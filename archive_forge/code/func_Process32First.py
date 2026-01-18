import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Process32First(hSnapshot):
    _Process32First = windll.kernel32.Process32First
    _Process32First.argtypes = [HANDLE, LPPROCESSENTRY32]
    _Process32First.restype = bool
    pe = PROCESSENTRY32()
    pe.dwSize = sizeof(PROCESSENTRY32)
    success = _Process32First(hSnapshot, byref(pe))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return pe