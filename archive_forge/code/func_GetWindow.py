from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindow(hWnd, uCmd):
    _GetWindow = windll.user32.GetWindow
    _GetWindow.argtypes = [HWND, UINT]
    _GetWindow.restype = HWND
    SetLastError(ERROR_SUCCESS)
    hWndTarget = _GetWindow(hWnd, uCmd)
    if not hWndTarget:
        winerr = GetLastError()
        if winerr != ERROR_SUCCESS:
            raise ctypes.WinError(winerr)
    return hWndTarget