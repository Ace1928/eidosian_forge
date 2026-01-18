import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class STORAGE_DEVICE_NUMBER(ctypes.Structure):
    _fields_ = [('DeviceType', DEVICE_TYPE), ('DeviceNumber', wintypes.DWORD), ('PartitionNumber', wintypes.DWORD)]