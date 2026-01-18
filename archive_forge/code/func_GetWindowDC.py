from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def GetWindowDC(hWnd):
    _GetWindowDC = windll.gdi32.GetWindowDC
    _GetWindowDC.argtypes = [HWND]
    _GetWindowDC.restype = HDC
    _GetWindowDC.errcheck = RaiseIfZero
    return _GetWindowDC(hWnd)