from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetClassNameW(hWnd):
    _GetClassNameW = windll.user32.GetClassNameW
    _GetClassNameW.argtypes = [HWND, LPWSTR, ctypes.c_int]
    _GetClassNameW.restype = ctypes.c_int
    nMaxCount = 4096
    dwCharSize = sizeof(WCHAR)
    while 1:
        lpClassName = ctypes.create_unicode_buffer(u'', nMaxCount)
        nCount = _GetClassNameW(hWnd, lpClassName, nMaxCount)
        if nCount == 0:
            raise ctypes.WinError()
        if nCount < nMaxCount - dwCharSize:
            break
        nMaxCount += 4096
    return lpClassName.value