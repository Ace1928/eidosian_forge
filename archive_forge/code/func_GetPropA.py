from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetPropA(hWnd, lpString):
    _GetPropA = windll.user32.GetPropA
    _GetPropA.argtypes = [HWND, LPSTR]
    _GetPropA.restype = HANDLE
    return _GetPropA(hWnd, lpString)