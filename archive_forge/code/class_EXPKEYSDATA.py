import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class EXPKEYSDATA(ctypes.Structure):
    _fields_ = (('nTablet', BYTE), ('nControl', BYTE), ('nLocation', BYTE), ('nReserved', BYTE), ('nState', DWORD))