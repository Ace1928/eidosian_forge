from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendMessageTimeoutA(hWnd, Msg, wParam=0, lParam=0, fuFlags=0, uTimeout=0):
    _SendMessageTimeoutA = windll.user32.SendMessageTimeoutA
    _SendMessageTimeoutA.argtypes = [HWND, UINT, WPARAM, LPARAM, UINT, UINT, PDWORD_PTR]
    _SendMessageTimeoutA.restype = LRESULT
    _SendMessageTimeoutA.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    dwResult = DWORD(0)
    _SendMessageTimeoutA(hWnd, Msg, wParam, lParam, fuFlags, uTimeout, byref(dwResult))
    return dwResult.value