import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class EXTENSIONBASE(ctypes.Structure):
    _fields_ = (('nContext', HCTX), ('nStatus', UINT), ('nTime', DWORD), ('nSerialNumber', UINT))