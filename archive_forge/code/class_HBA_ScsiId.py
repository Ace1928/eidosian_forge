import ctypes
import os_win.conf
from os_win.utils.winapi import wintypes
class HBA_ScsiId(ctypes.Structure):
    _fields_ = [('OSDeviceName', wintypes.CHAR * 256), ('ScsiBusNumber', ctypes.c_uint32), ('ScsiTargetNumber', ctypes.c_uint32), ('ScsiOSLun', ctypes.c_uint32)]