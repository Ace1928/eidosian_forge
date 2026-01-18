from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def MapWindowPoints(hWndFrom, hWndTo, lpPoints):
    _MapWindowPoints = windll.user32.MapWindowPoints
    _MapWindowPoints.argtypes = [HWND, HWND, LPPOINT, UINT]
    _MapWindowPoints.restype = ctypes.c_int
    cPoints = len(lpPoints)
    lpPoints = (POINT * cPoints)(*lpPoints)
    SetLastError(ERROR_SUCCESS)
    number = _MapWindowPoints(hWndFrom, hWndTo, byref(lpPoints), cPoints)
    if number == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    x_delta = number & 65535
    y_delta = number >> 16 & 65535
    return (x_delta, y_delta, [(Point.x, Point.y) for Point in lpPoints])