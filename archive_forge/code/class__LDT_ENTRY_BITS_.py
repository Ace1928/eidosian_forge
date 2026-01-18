from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class _LDT_ENTRY_BITS_(Structure):
    _pack_ = 1
    _fields_ = [('BaseMid', DWORD, 8), ('Type', DWORD, 5), ('Dpl', DWORD, 2), ('Pres', DWORD, 1), ('LimitHi', DWORD, 4), ('Sys', DWORD, 1), ('Reserved_0', DWORD, 1), ('Default_Big', DWORD, 1), ('Granularity', DWORD, 1), ('BaseHi', DWORD, 8)]