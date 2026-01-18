import ctypes
from os_win.utils.winapi import wintypes
class CLUSTER_ENUM_ITEM(ctypes.Structure):
    _fields_ = [('dwVersion', wintypes.DWORD), ('dwType', wintypes.DWORD), ('cbId', wintypes.DWORD), ('lpszId', wintypes.LPWSTR), ('cbName', wintypes.DWORD), ('lpszName', wintypes.LPWSTR)]