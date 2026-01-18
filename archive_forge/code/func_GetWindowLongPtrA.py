from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowLongPtrA(hWnd, nIndex=0):
    _GetWindowLongPtrA = windll.user32.GetWindowLongPtrA
    _GetWindowLongPtrA.argtypes = [HWND, ctypes.c_int]
    _GetWindowLongPtrA.restype = SIZE_T
    SetLastError(ERROR_SUCCESS)
    retval = _GetWindowLongPtrA(hWnd, nIndex)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval