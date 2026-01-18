from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def EnumThreadWindows(dwThreadId):
    _EnumThreadWindows = windll.user32.EnumThreadWindows
    _EnumThreadWindows.argtypes = [DWORD, WNDENUMPROC, LPARAM]
    _EnumThreadWindows.restype = bool
    fn = __EnumThreadWndProc()
    lpfn = WNDENUMPROC(fn)
    if not _EnumThreadWindows(dwThreadId, lpfn, NULL):
        errcode = GetLastError()
        if errcode not in (ERROR_NO_MORE_FILES, ERROR_SUCCESS):
            raise ctypes.WinError(errcode)
    return fn.hwnd