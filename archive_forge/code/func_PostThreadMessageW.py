from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def PostThreadMessageW(idThread, Msg, wParam=0, lParam=0):
    _PostThreadMessageW = windll.user32.PostThreadMessageW
    _PostThreadMessageW.argtypes = [DWORD, UINT, WPARAM, LPARAM]
    _PostThreadMessageW.restype = bool
    _PostThreadMessageW.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostThreadMessageW(idThread, Msg, wParam, lParam)