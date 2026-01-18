from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowLongPtrW(hWnd, nIndex, dwNewLong):
    _SetWindowLongPtrW = windll.user32.SetWindowLongPtrW
    _SetWindowLongPtrW.argtypes = [HWND, ctypes.c_int, SIZE_T]
    _SetWindowLongPtrW.restype = SIZE_T
    SetLastError(ERROR_SUCCESS)
    retval = _SetWindowLongPtrW(hWnd, nIndex, dwNewLong)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval