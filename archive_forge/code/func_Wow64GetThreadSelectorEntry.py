from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
def Wow64GetThreadSelectorEntry(hThread, dwSelector):
    _Wow64GetThreadSelectorEntry = windll.kernel32.Wow64GetThreadSelectorEntry
    _Wow64GetThreadSelectorEntry.argtypes = [HANDLE, DWORD, PWOW64_LDT_ENTRY]
    _Wow64GetThreadSelectorEntry.restype = bool
    _Wow64GetThreadSelectorEntry.errcheck = RaiseIfZero
    lpSelectorEntry = WOW64_LDT_ENTRY()
    _Wow64GetThreadSelectorEntry(hThread, dwSelector, byref(lpSelectorEntry))
    return lpSelectorEntry