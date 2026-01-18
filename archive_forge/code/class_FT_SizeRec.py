from ctypes import *
from .base import FontException
import pyglet.lib
class FT_SizeRec(Structure):
    _fields_ = [('face', c_void_p), ('generic', FT_Generic), ('metrics', FT_Size_Metrics), ('internal', c_void_p)]