from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowRect(hWnd):
    _GetWindowRect = windll.user32.GetWindowRect
    _GetWindowRect.argtypes = [HWND, LPRECT]
    _GetWindowRect.restype = bool
    _GetWindowRect.errcheck = RaiseIfZero
    lpRect = RECT()
    _GetWindowRect(hWnd, byref(lpRect))
    return Rect(lpRect.left, lpRect.top, lpRect.right, lpRect.bottom)