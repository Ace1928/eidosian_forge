import ctypes
from os_win.utils.winapi import wintypes
class NOTIFY_FILTER_AND_TYPE(ctypes.Structure):
    _fields_ = [('dwObjectType', wintypes.DWORD), ('FilterFlags', wintypes.LONGLONG)]