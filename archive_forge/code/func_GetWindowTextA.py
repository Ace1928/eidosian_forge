from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowTextA(hWnd):
    _GetWindowTextA = windll.user32.GetWindowTextA
    _GetWindowTextA.argtypes = [HWND, LPSTR, ctypes.c_int]
    _GetWindowTextA.restype = ctypes.c_int
    nMaxCount = 4096
    dwCharSize = sizeof(CHAR)
    while 1:
        lpString = ctypes.create_string_buffer('', nMaxCount)
        nCount = _GetWindowTextA(hWnd, lpString, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 4096
    return lpString.value