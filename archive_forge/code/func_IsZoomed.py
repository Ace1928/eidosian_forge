from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsZoomed(hWnd):
    _IsZoomed = windll.user32.IsZoomed
    _IsZoomed.argtypes = [HWND]
    _IsZoomed.restype = bool
    return _IsZoomed(hWnd)