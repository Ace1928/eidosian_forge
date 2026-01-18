import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
def MAKEINTRESOURCE(i):
    return cast(ctypes.c_void_p(i & 65535), c_wchar_p)