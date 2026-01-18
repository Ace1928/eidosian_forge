import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class PACKETEXT(ctypes.Structure):
    _fields_ = (('pkBase', EXTENSIONBASE), ('pkExpKeys', EXPKEYSDATA), ('pkTouchStrip', SLIDERDATA), ('pkTouchRing', SLIDERDATA))