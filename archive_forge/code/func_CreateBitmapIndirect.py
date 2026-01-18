from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def CreateBitmapIndirect(lpbm):
    _CreateBitmapIndirect = windll.gdi32.CreateBitmapIndirect
    _CreateBitmapIndirect.argtypes = [PBITMAP]
    _CreateBitmapIndirect.restype = HBITMAP
    _CreateBitmapIndirect.errcheck = RaiseIfZero
    return _CreateBitmapIndirect(lpbm)