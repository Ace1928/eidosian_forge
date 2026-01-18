import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class JIT_DEBUG_INFO(Structure):
    _fields_ = [('dwSize', DWORD), ('dwProcessorArchitecture', DWORD), ('dwThreadID', DWORD), ('dwReserved0', DWORD), ('lpExceptionAddress', ULONG64), ('lpExceptionRecord', ULONG64), ('lpContextRecord', ULONG64)]