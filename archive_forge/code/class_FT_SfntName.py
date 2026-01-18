from ctypes import *
from .base import FontException
import pyglet.lib
class FT_SfntName(Structure):
    _fields_ = [('platform_id', FT_UShort), ('encoding_id', FT_UShort), ('language_id', FT_UShort), ('name_id', FT_UShort), ('string', POINTER(FT_Byte)), ('string_len', FT_UInt)]