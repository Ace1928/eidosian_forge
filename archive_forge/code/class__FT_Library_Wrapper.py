import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class _FT_Library_Wrapper(FT_Library):
    """Subclass of FT_Library to help with calling FT_Done_FreeType"""
    _type_ = FT_Library._type_
    _ft_done_freetype = FT_Done_FreeType

    def __del__(self):
        pass