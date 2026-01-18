import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class STATSTG(Structure):
    _fields_ = [('pwcsName', LPOLESTR), ('type', DWORD), ('cbSize', ULARGE_INTEGER), ('mtime', FILETIME), ('ctime', FILETIME), ('atime', FILETIME), ('grfMode', DWORD), ('grfLocksSupported', DWORD), ('clsid', DWORD), ('grfStateBits', DWORD), ('reserved', DWORD)]