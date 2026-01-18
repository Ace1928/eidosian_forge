from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowPlacement(hWnd, lpwndpl):
    _SetWindowPlacement = windll.user32.SetWindowPlacement
    _SetWindowPlacement.argtypes = [HWND, PWINDOWPLACEMENT]
    _SetWindowPlacement.restype = bool
    _SetWindowPlacement.errcheck = RaiseIfZero
    if isinstance(lpwndpl, WINDOWPLACEMENT):
        lpwndpl.length = sizeof(lpwndpl)
    _SetWindowPlacement(hWnd, byref(lpwndpl))