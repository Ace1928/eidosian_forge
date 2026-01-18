import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetProcessAffinityMask(hProcess, dwProcessAffinityMask):
    _SetProcessAffinityMask = windll.kernel32.SetProcessAffinityMask
    _SetProcessAffinityMask.argtypes = [HANDLE, DWORD_PTR]
    _SetProcessAffinityMask.restype = bool
    _SetProcessAffinityMask.errcheck = RaiseIfZero
    _SetProcessAffinityMask(hProcess, dwProcessAffinityMask)