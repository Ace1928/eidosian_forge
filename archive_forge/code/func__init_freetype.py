import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _init_freetype():
    global _handle
    _handle = _FT_Library_Wrapper()
    error = FT_Init_FreeType(byref(_handle))
    if error:
        raise FT_Exception(error)
    try:
        set_lcd_filter(FT_LCD_FILTER_DEFAULT)
    except:
        pass