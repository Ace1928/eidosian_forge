import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class PACKET(ctypes.Structure):
    _fields_ = (('pkChanged', WTPKT), ('pkCursor', UINT), ('pkButtons', DWORD), ('pkX', LONG), ('pkY', LONG), ('pkZ', LONG), ('pkNormalPressure', UINT), ('pkTangentPressure', UINT), ('pkOrientation', ORIENTATION))