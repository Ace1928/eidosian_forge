import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class ICONINFO(Structure):
    _fields_ = [('fIcon', BOOL), ('xHotspot', DWORD), ('yHotspot', DWORD), ('hbmMask', HBITMAP), ('hbmColor', HBITMAP)]
    __slots__ = [f[0] for f in _fields_]