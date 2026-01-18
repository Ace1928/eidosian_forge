import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def _scale_gesture(self):
    """ Scales down the gesture to a unit of 1."""
    min_x = min([stroke.min_x for stroke in self.strokes])
    max_x = max([stroke.max_x for stroke in self.strokes])
    min_y = min([stroke.min_y for stroke in self.strokes])
    max_y = max([stroke.max_y for stroke in self.strokes])
    x_len = max_x - min_x
    self.width = x_len
    y_len = max_y - min_y
    self.height = y_len
    scale_factor = max(x_len, y_len)
    if scale_factor <= 0.0:
        return False
    scale_factor = 1.0 / scale_factor
    for stroke in self.strokes:
        stroke.scale_stroke(scale_factor)
    return True