from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowLongPtrA(hWnd, nIndex, dwNewLong):
    _SetWindowLongPtrA = windll.user32.SetWindowLongPtrA
    _SetWindowLongPtrA.argtypes = [HWND, ctypes.c_int, SIZE_T]
    _SetWindowLongPtrA.restype = SIZE_T
    SetLastError(ERROR_SUCCESS)
    retval = _SetWindowLongPtrA(hWnd, nIndex, dwNewLong)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval