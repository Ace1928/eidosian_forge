import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class RAWHID(Structure):
    _fields_ = [('dwSizeHid', DWORD), ('dwCount', DWORD), ('bRawData', POINTER(BYTE))]