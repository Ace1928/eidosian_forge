import sys
import cv2 as cv
@register('cv2')
def GIn(*args):
    return [*args]