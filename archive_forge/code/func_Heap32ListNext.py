import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Heap32ListNext(hSnapshot, hl=None):
    _Heap32ListNext = windll.kernel32.Heap32ListNext
    _Heap32ListNext.argtypes = [HANDLE, LPHEAPLIST32]
    _Heap32ListNext.restype = bool
    if hl is None:
        hl = HEAPLIST32()
    hl.dwSize = sizeof(HEAPLIST32)
    success = _Heap32ListNext(hSnapshot, byref(hl))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return hl