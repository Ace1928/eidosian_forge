import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class DeviceInterfaceDetailData(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('DevicePath', ctypes.c_byte * 1)]
    _pack_ = SETUPAPI_PACK