import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class MEMORY_BASIC_INFORMATION(Structure):
    _fields_ = [('BaseAddress', SIZE_T), ('AllocationBase', SIZE_T), ('AllocationProtect', DWORD), ('RegionSize', SIZE_T), ('State', DWORD), ('Protect', DWORD), ('Type', DWORD)]