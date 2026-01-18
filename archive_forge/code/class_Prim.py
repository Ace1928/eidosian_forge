import sys
import cv2 as cv
class Prim:

    def __new__(self):
        return cv.GArray(cv.gapi.CV_DRAW_PRIM)