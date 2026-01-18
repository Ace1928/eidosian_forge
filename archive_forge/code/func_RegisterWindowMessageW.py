from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def RegisterWindowMessageW(lpString):
    _RegisterWindowMessageW = windll.user32.RegisterWindowMessageW
    _RegisterWindowMessageW.argtypes = [LPWSTR]
    _RegisterWindowMessageW.restype = UINT
    _RegisterWindowMessageW.errcheck = RaiseIfZero
    return _RegisterWindowMessageW(lpString)