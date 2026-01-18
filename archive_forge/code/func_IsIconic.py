from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsIconic(hWnd):
    _IsIconic = windll.user32.IsIconic
    _IsIconic.argtypes = [HWND]
    _IsIconic.restype = bool
    return _IsIconic(hWnd)