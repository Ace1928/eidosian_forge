from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def GetObjectType(h):
    _GetObjectType = windll.gdi32.GetObjectType
    _GetObjectType.argtypes = [HGDIOBJ]
    _GetObjectType.restype = DWORD
    _GetObjectType.errcheck = RaiseIfZero
    return _GetObjectType(h)