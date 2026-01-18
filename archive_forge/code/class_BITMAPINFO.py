import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class BITMAPINFO(Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', RGBQUAD * 1)]
    __slots__ = [f[0] for f in _fields_]