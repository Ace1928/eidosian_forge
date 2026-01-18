from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def RemovePropW(hWnd, lpString):
    _RemovePropW = windll.user32.RemovePropW
    _RemovePropW.argtypes = [HWND, LPWSTR]
    _RemovePropW.restype = HANDLE
    return _RemovePropW(hWnd, lpString)