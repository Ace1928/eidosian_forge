import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Heap32Next(he):
    _Heap32Next = windll.kernel32.Heap32Next
    _Heap32Next.argtypes = [LPHEAPENTRY32]
    _Heap32Next.restype = bool
    he.dwSize = sizeof(HEAPENTRY32)
    success = _Heap32Next(byref(he))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return he