from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def WaitForInputIdle(hProcess, dwMilliseconds=INFINITE):
    _WaitForInputIdle = windll.user32.WaitForInputIdle
    _WaitForInputIdle.argtypes = [HANDLE, DWORD]
    _WaitForInputIdle.restype = DWORD
    r = _WaitForInputIdle(hProcess, dwMilliseconds)
    if r == WAIT_FAILED:
        raise ctypes.WinError()
    return r