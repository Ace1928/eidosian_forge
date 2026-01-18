import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class CIEXYZ(Structure):
    _fields_ = [('ciexyzX', DWORD), ('ciexyzY', DWORD), ('ciexyzZ', DWORD)]
    __slots__ = [f[0] for f in _fields_]