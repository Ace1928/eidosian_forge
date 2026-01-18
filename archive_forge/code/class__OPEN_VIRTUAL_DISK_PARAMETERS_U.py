import ctypes
from os_win.utils.winapi import wintypes
class _OPEN_VIRTUAL_DISK_PARAMETERS_U(ctypes.Union):
    _fields_ = [('Version1', _OPEN_VIRTUAL_DISK_PARAMETERS_V1), ('Version2', _OPEN_VIRTUAL_DISK_PARAMETERS_V2)]