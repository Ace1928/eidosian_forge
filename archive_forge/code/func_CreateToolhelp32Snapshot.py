import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateToolhelp32Snapshot(dwFlags=TH32CS_SNAPALL, th32ProcessID=0):
    _CreateToolhelp32Snapshot = windll.kernel32.CreateToolhelp32Snapshot
    _CreateToolhelp32Snapshot.argtypes = [DWORD, DWORD]
    _CreateToolhelp32Snapshot.restype = HANDLE
    hSnapshot = _CreateToolhelp32Snapshot(dwFlags, th32ProcessID)
    if hSnapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError()
    return SnapshotHandle(hSnapshot)