import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class MINMAXINFO(Structure):
    _fields_ = [('ptReserved', POINT), ('ptMaxSize', POINT), ('ptMaxPosition', POINT), ('ptMinTrackSize', POINT), ('ptMaxTrackSize', POINT)]
    __slots__ = [f[0] for f in _fields_]