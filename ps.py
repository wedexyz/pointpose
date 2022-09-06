import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.
mp_pose = mp.solutions.pose


frameWidth = 800
frameHeight = 640
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
deadZone=100
cv2.resizeWindow("HSV",800,640)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        _, img = cap.read()
        image  = img.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS )
        '''
        print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frameWidth}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frameHeight})'
        )
       
        '''
        print(mp_pose.POSE_CONNECTIONS)
        #cv2.line(image,(int(frameWidth/2)-deadZone,0),(int(frameWidth/2)-deadZone,frameHeight),(255,255,0),3)
        #cv2.line(image,(int(frameWidth/2)+deadZone,0),(int(frameWidth/2)+deadZone,frameHeight),(255,255,0),3)
        #cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5)
        #cv2.line(image, (0,int(frameHeight / 2) - deadZone), (frameWidth,int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
        #cv2.line(image, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)

        cv2.imshow('Horizontal Stacking', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()