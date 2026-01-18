import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class ROTATION(ctypes.Structure):
    _fields_ = (('roPitch', ctypes.c_int), ('roRoll', ctypes.c_int), ('roYaw', ctypes.c_int))