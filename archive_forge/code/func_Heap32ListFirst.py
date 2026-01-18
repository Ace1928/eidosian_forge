import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Heap32ListFirst(hSnapshot):
    _Heap32ListFirst = windll.kernel32.Heap32ListFirst
    _Heap32ListFirst.argtypes = [HANDLE, LPHEAPLIST32]
    _Heap32ListFirst.restype = bool
    hl = HEAPLIST32()
    hl.dwSize = sizeof(HEAPLIST32)
    success = _Heap32ListFirst(hSnapshot, byref(hl))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return hl