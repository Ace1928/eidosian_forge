import ctypes
from os_win.utils.winapi import wintypes
class GET_VIRTUAL_DISK_INFO(ctypes.Structure):
    _anonymous_ = ['_vhdinfo']
    _fields_ = [('Version', wintypes.UINT), ('_vhdinfo', _VHD_INFO)]