from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def GetBitmapBits(hbmp):
    _GetBitmapBits = windll.gdi32.GetBitmapBits
    _GetBitmapBits.argtypes = [HBITMAP, LONG, LPVOID]
    _GetBitmapBits.restype = LONG
    _GetBitmapBits.errcheck = RaiseIfZero
    bitmap = GetObject(hbmp, lpvObject=BITMAP())
    cbBuffer = bitmap.bmWidthBytes * bitmap.bmHeight
    lpvBits = ctypes.create_string_buffer('', cbBuffer)
    _GetBitmapBits(hbmp, cbBuffer, byref(lpvBits))
    return lpvBits.raw