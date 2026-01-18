import ctypes
import ctypes.wintypes
import stat as stdstat
from collections import namedtuple
class FILETIME(ctypes.Structure):
    _fields_ = [('dwLowDateTime', ctypes.wintypes.DWORD), ('dwHighDateTime', ctypes.wintypes.DWORD)]