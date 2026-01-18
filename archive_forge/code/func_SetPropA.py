from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetPropA(hWnd, lpString, hData):
    _SetPropA = windll.user32.SetPropA
    _SetPropA.argtypes = [HWND, LPSTR, HANDLE]
    _SetPropA.restype = BOOL
    _SetPropA.errcheck = RaiseIfZero
    _SetPropA(hWnd, lpString, hData)