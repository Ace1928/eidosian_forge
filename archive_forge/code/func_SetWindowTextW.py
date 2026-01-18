from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowTextW(hWnd, lpString=None):
    _SetWindowTextW = windll.user32.SetWindowTextW
    _SetWindowTextW.argtypes = [HWND, LPWSTR]
    _SetWindowTextW.restype = bool
    _SetWindowTextW.errcheck = RaiseIfZero
    _SetWindowTextW(hWnd, lpString)