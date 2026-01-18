import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _DUMMYSTRUCTNAME2(Structure):
    _fields_ = [('dmPosition', POINTL), ('dmDisplayOrientation', DWORD), ('dmDisplayFixedOutput', DWORD)]