from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsWindowEnabled(hWnd):
    _IsWindowEnabled = windll.user32.IsWindowEnabled
    _IsWindowEnabled.argtypes = [HWND]
    _IsWindowEnabled.restype = bool
    return _IsWindowEnabled(hWnd)