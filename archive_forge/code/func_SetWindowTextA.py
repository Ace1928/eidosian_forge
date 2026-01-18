from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowTextA(hWnd, lpString=None):
    _SetWindowTextA = windll.user32.SetWindowTextA
    _SetWindowTextA.argtypes = [HWND, LPSTR]
    _SetWindowTextA.restype = bool
    _SetWindowTextA.errcheck = RaiseIfZero
    _SetWindowTextA(hWnd, lpString)