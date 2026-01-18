import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class EXTPROPERTY(ctypes.Structure):
    _fields_ = (('version', BYTE), ('tabletIndex', BYTE), ('controlIndex', BYTE), ('functionIndex', BYTE), ('propertyID', WORD), ('reserved', WORD), ('dataSize', DWORD), ('data', BYTE * 1))