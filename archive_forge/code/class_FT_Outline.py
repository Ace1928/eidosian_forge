from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Outline(Structure):
    _fields_ = [('n_contours', c_short), ('n_points', c_short), ('points', POINTER(FT_Vector)), ('tags', c_char_p), ('contours', POINTER(c_short)), ('flags', c_int)]