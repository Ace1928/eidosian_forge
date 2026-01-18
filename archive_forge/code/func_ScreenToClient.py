from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def ScreenToClient(hWnd, lpPoint):
    _ScreenToClient = windll.user32.ScreenToClient
    _ScreenToClient.argtypes = [HWND, LPPOINT]
    _ScreenToClient.restype = bool
    _ScreenToClient.errcheck = RaiseIfZero
    if isinstance(lpPoint, tuple):
        lpPoint = POINT(*lpPoint)
    else:
        lpPoint = POINT(lpPoint.x, lpPoint.y)
    _ScreenToClient(hWnd, byref(lpPoint))
    return Point(lpPoint.x, lpPoint.y)