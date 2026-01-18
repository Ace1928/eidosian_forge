import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class PROPVARIANT(Structure):
    _anonymous_ = ['union']
    _fields_ = [('vt', ctypes.c_ushort), ('wReserved1', ctypes.c_ubyte), ('wReserved2', ctypes.c_ubyte), ('wReserved3', ctypes.c_ulong), ('union', _VarTable)]