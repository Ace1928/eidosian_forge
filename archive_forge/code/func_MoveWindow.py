from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def MoveWindow(hWnd, X, Y, nWidth, nHeight, bRepaint=True):
    _MoveWindow = windll.user32.MoveWindow
    _MoveWindow.argtypes = [HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, BOOL]
    _MoveWindow.restype = bool
    _MoveWindow.errcheck = RaiseIfZero
    _MoveWindow(hWnd, X, Y, nWidth, nHeight, bool(bRepaint))