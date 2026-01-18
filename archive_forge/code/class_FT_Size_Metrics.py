from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Size_Metrics(Structure):
    _fields_ = [('x_ppem', FT_UShort), ('y_ppem', FT_UShort), ('x_scale', FT_Fixed), ('y_scale', FT_Fixed), ('ascender', FT_Pos), ('descender', FT_Pos), ('height', FT_Pos), ('max_advance', FT_Pos)]