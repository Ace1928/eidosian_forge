from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetPropW(hWnd, lpString):
    _GetPropW = windll.user32.GetPropW
    _GetPropW.argtypes = [HWND, LPWSTR]
    _GetPropW.restype = HANDLE
    return _GetPropW(hWnd, lpString)