from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsChild(hWnd):
    _IsChild = windll.user32.IsChild
    _IsChild.argtypes = [HWND]
    _IsChild.restype = bool
    return _IsChild(hWnd)