import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Thread32Next(hSnapshot, te=None):
    _Thread32Next = windll.kernel32.Thread32Next
    _Thread32Next.argtypes = [HANDLE, LPTHREADENTRY32]
    _Thread32Next.restype = bool
    if te is None:
        te = THREADENTRY32()
    te.dwSize = sizeof(THREADENTRY32)
    success = _Thread32Next(hSnapshot, byref(te))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return te