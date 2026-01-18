import ctypes
from os_win.utils.winapi import wintypes
class CREATE_VIRTUAL_DISK_PARAMETERS(ctypes.Structure):
    _fields_ = [('Version', wintypes.DWORD), ('Version2', _CREATE_VIRTUAL_DISK_PARAMETERS_V2)]