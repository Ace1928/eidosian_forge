from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Bitmap(Structure):
    _fields_ = [('rows', c_uint), ('width', c_uint), ('pitch', c_int), ('buffer', POINTER(c_ubyte)), ('num_grays', c_short), ('pixel_mode', c_ubyte), ('palette_mode', c_ubyte), ('palette', c_void_p)]