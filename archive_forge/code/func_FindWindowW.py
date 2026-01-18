from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def FindWindowW(lpClassName=None, lpWindowName=None):
    _FindWindowW = windll.user32.FindWindowW
    _FindWindowW.argtypes = [LPWSTR, LPWSTR]
    _FindWindowW.restype = HWND
    hWnd = _FindWindowW(lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd