import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class SLIDERDATA(ctypes.Structure):
    _fields_ = (('nTablet', BYTE), ('nControl', BYTE), ('nMode', BYTE), ('nReserved', BYTE), ('nPosition', DWORD))