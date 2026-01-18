from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Vector(Structure):
    _fields_ = [('x', FT_Pos), ('y', FT_Pos)]