import sys
import cv2 as cv
class Int64:

    def __new__(self):
        return cv.GArrayT(cv.gapi.CV_INT64)