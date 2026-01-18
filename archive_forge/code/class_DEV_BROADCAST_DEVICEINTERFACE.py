import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class DEV_BROADCAST_DEVICEINTERFACE(Structure):
    _fields_ = (('dbcc_size', DWORD), ('dbcc_devicetype', DWORD), ('dbcc_reserved', DWORD), ('dbcc_classguid', com.GUID), ('dbcc_name', ctypes.c_wchar * 256))