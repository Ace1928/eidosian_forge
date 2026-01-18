from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowLongA(hWnd, nIndex=0):
    _GetWindowLongA = windll.user32.GetWindowLongA
    _GetWindowLongA.argtypes = [HWND, ctypes.c_int]
    _GetWindowLongA.restype = DWORD
    SetLastError(ERROR_SUCCESS)
    retval = _GetWindowLongA(hWnd, nIndex)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval