from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetAncestor(hWnd, gaFlags=GA_PARENT):
    _GetAncestor = windll.user32.GetAncestor
    _GetAncestor.argtypes = [HWND, UINT]
    _GetAncestor.restype = HWND
    SetLastError(ERROR_SUCCESS)
    hWndParent = _GetAncestor(hWnd, gaFlags)
    if not hWndParent:
        winerr = GetLastError()
        if winerr != ERROR_SUCCESS:
            raise ctypes.WinError(winerr)
    return hWndParent