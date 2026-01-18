from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetClientRect(hWnd):
    _GetClientRect = windll.user32.GetClientRect
    _GetClientRect.argtypes = [HWND, LPRECT]
    _GetClientRect.restype = bool
    _GetClientRect.errcheck = RaiseIfZero
    lpRect = RECT()
    _GetClientRect(hWnd, byref(lpRect))
    return Rect(lpRect.left, lpRect.top, lpRect.right, lpRect.bottom)