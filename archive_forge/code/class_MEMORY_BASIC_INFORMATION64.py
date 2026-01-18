import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class MEMORY_BASIC_INFORMATION64(Structure):
    _fields_ = [('BaseAddress', ULONGLONG), ('AllocationBase', ULONGLONG), ('AllocationProtect', DWORD), ('__alignment1', DWORD), ('RegionSize', ULONGLONG), ('State', DWORD), ('Protect', DWORD), ('Type', DWORD), ('__alignment2', DWORD)]