import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Heap32First(th32ProcessID, th32HeapID):
    _Heap32First = windll.kernel32.Heap32First
    _Heap32First.argtypes = [LPHEAPENTRY32, DWORD, ULONG_PTR]
    _Heap32First.restype = bool
    he = HEAPENTRY32()
    he.dwSize = sizeof(HEAPENTRY32)
    success = _Heap32First(byref(he), th32ProcessID, th32HeapID)
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return he