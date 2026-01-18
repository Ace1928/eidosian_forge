import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def FT_Done_MM_Var_func(p):
    error = FT_Done_MM_Var(get_handle(), p)
    if error:
        raise FT_Exception('Failure calling FT_Done_MM_Var')