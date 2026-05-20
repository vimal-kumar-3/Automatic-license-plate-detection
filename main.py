import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.title("Automatic License Plate Detection System")

# Create temporary folder
os.makedirs("temp", exist_ok=True)

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Load YOLO model
try:
    model = YOLO(r'D:\Main_Desktop\automatic-license-plate-detection-project-yolo\best_license_plate_model.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")


# ---------------- IMAGE FUNCTION ----------------
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model.predict(image, device='cpu')

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{confidence*100:.2f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    return image


# ---------------- LIVE VIDEO FUNCTION ----------------
def live_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file")
        return

    stframe = st.empty()   # LIVE display container

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device='cpu')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{confidence*100:.2f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # SHOW LIVE FRAME
        stframe.image(frame, channels="RGB")

    cap.release()


# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    input_path = os.path.join("temp", uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = os.path.splitext(input_path)[1].lower()

    st.write("Processing...")

    # IMAGE
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        result_image = predict_image(input_path)
        st.image(result_image, caption="Detected Image")

    # VIDEO (LIVE)
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        live_video_detection(input_path)

    else:
        st.error("Unsupported file type")

       