import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Thread32First(hSnapshot):
    _Thread32First = windll.kernel32.Thread32First
    _Thread32First.argtypes = [HANDLE, LPTHREADENTRY32]
    _Thread32First.restype = bool
    te = THREADENTRY32()
    te.dwSize = sizeof(THREADENTRY32)
    success = _Thread32First(hSnapshot, byref(te))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return te