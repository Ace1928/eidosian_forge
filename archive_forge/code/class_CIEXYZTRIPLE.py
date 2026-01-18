import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class CIEXYZTRIPLE(Structure):
    _fields_ = [('ciexyzRed', CIEXYZ), ('ciexyzBlue', CIEXYZ), ('ciexyzGreen', CIEXYZ)]
    __slots__ = [f[0] for f in _fields_]