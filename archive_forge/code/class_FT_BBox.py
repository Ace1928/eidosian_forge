from ctypes import *
from .base import FontException
import pyglet.lib
class FT_BBox(Structure):
    _fields_ = [('xMin', FT_Pos), ('yMin', FT_Pos), ('xMax', FT_Pos), ('yMax', FT_Pos)]