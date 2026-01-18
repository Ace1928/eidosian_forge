from ctypes import c_void_p
from ctypes import Structure
from ctypes import Union
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
class _DummyStruct(Structure):
    _fields_ = [('Offset', DWORD), ('OffsetHigh', DWORD)]