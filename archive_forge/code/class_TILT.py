import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class TILT(ctypes.Structure):
    _fields_ = (('tiltX', ctypes.c_int), ('tiltY', ctypes.c_int))