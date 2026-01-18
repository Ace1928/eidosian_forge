from __future__ import absolute_import
from ctypes import c_ulong, c_void_p, c_int64, c_char, \
from ctypes.wintypes import HANDLE
from ctypes.wintypes import BOOL
from ctypes.wintypes import LPCWSTR
from ctypes.wintypes import DWORD
from ctypes.wintypes import WORD
from ctypes.wintypes import BYTE
class _SECURITY_ATTRIBUTES(Structure):
    pass