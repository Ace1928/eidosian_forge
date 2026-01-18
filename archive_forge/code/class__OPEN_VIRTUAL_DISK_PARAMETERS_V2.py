import ctypes
from os_win.utils.winapi import wintypes
class _OPEN_VIRTUAL_DISK_PARAMETERS_V2(ctypes.Structure):
    _fields_ = [('GetInfoOnly', wintypes.BOOL), ('ReadOnly', wintypes.BOOL), ('ResiliencyGuid', wintypes.GUID)]