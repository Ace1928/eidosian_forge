from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def FindWindowExA(hwndParent=None, hwndChildAfter=None, lpClassName=None, lpWindowName=None):
    _FindWindowExA = windll.user32.FindWindowExA
    _FindWindowExA.argtypes = [HWND, HWND, LPSTR, LPSTR]
    _FindWindowExA.restype = HWND
    hWnd = _FindWindowExA(hwndParent, hwndChildAfter, lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd