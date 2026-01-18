import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class RAWINPUTDEVICE(Structure):
    _fields_ = [('usUsagePage', USHORT), ('usUsage', USHORT), ('dwFlags', DWORD), ('hwndTarget', HWND)]