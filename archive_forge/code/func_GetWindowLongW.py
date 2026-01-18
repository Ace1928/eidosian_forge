from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowLongW(hWnd, nIndex=0):
    _GetWindowLongW = windll.user32.GetWindowLongW
    _GetWindowLongW.argtypes = [HWND, ctypes.c_int]
    _GetWindowLongW.restype = DWORD
    SetLastError(ERROR_SUCCESS)
    retval = _GetWindowLongW(hWnd, nIndex)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval