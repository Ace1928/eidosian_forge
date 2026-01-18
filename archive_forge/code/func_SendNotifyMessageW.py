from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendNotifyMessageW(hWnd, Msg, wParam=0, lParam=0):
    _SendNotifyMessageW = windll.user32.SendNotifyMessageW
    _SendNotifyMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendNotifyMessageW.restype = bool
    _SendNotifyMessageW.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _SendNotifyMessageW(hWnd, Msg, wParam, lParam)