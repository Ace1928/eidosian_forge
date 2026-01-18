import ctypes
from os_win.utils.winapi import wintypes
class OPEN_VIRTUAL_DISK_PARAMETERS(ctypes.Structure):
    _anonymous_ = ['_parameters']
    _fields_ = [('Version', wintypes.DWORD), ('_parameters', _OPEN_VIRTUAL_DISK_PARAMETERS_U)]