from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Data(Structure):
    _fields_ = [('pointer', POINTER(FT_Byte)), ('length', FT_Int)]