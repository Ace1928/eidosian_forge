import ctypes
from os_win.utils.winapi import wintypes
class _SET_VIRTUAL_DISK_INFO_U(ctypes.Union):
    _fields_ = [('ParentFilePath', wintypes.LPCWSTR), ('UniqueIdentifier', wintypes.GUID), ('VirtualDiskId', wintypes.GUID)]