from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetLastErrorEx(dwErrCode, dwType=0):
    _SetLastErrorEx = windll.user32.SetLastErrorEx
    _SetLastErrorEx.argtypes = [DWORD, DWORD]
    _SetLastErrorEx.restype = None
    _SetLastErrorEx(dwErrCode, dwType)