import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class MONITORINFOEX(Structure):
    _fields_ = [('cbSize', DWORD), ('rcMonitor', RECT), ('rcWork', RECT), ('dwFlags', DWORD), ('szDevice', WCHAR * CCHDEVICENAME)]
    __slots__ = [f[0] for f in _fields_]