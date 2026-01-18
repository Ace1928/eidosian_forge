from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def GetDC(hWnd):
    _GetDC = windll.gdi32.GetDC
    _GetDC.argtypes = [HWND]
    _GetDC.restype = HDC
    _GetDC.errcheck = RaiseIfZero
    return _GetDC(hWnd)