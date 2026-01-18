import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi
class COPYDATASTRUCT(ctypes.Structure):
    """
    ctypes implementation of
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms649010%28v=vs.85%29.aspx
    """
    _fields_ = [('num_data', ULONG_PTR), ('data_size', ctypes.wintypes.DWORD), ('data_loc', ctypes.c_void_p)]