import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class TRACKMOUSEEVENT(Structure):
    _fields_ = [('cbSize', DWORD), ('dwFlags', DWORD), ('hwndTrack', HWND), ('dwHoverTime', DWORD)]
    __slots__ = [f[0] for f in _fields_]