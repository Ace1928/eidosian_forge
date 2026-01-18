from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def EnumWindows():
    _EnumWindows = windll.user32.EnumWindows
    _EnumWindows.argtypes = [WNDENUMPROC, LPARAM]
    _EnumWindows.restype = bool
    EnumFunc = __EnumWndProc()
    lpEnumFunc = WNDENUMPROC(EnumFunc)
    if not _EnumWindows(lpEnumFunc, NULL):
        errcode = GetLastError()
        if errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
            raise ctypes.WinError(errcode)
    return EnumFunc.hwnd