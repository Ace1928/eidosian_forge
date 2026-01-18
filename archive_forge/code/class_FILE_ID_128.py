import ctypes
from os_win.utils.winapi import wintypes
class FILE_ID_128(ctypes.Structure):
    _fields_ = [('Identifier', wintypes.BYTE * 16)]