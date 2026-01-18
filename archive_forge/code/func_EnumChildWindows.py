from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def EnumChildWindows(hWndParent=NULL):
    _EnumChildWindows = windll.user32.EnumChildWindows
    _EnumChildWindows.argtypes = [HWND, WNDENUMPROC, LPARAM]
    _EnumChildWindows.restype = bool
    EnumFunc = __EnumChildProc()
    lpEnumFunc = WNDENUMPROC(EnumFunc)
    SetLastError(ERROR_SUCCESS)
    _EnumChildWindows(hWndParent, lpEnumFunc, NULL)
    errcode = GetLastError()
    if errcode != ERROR_SUCCESS and errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
        raise ctypes.WinError(errcode)
    return EnumFunc.hwnd