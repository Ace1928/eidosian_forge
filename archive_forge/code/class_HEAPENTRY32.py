import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class HEAPENTRY32(Structure):
    _fields_ = [('dwSize', SIZE_T), ('hHandle', HANDLE), ('dwAddress', ULONG_PTR), ('dwBlockSize', SIZE_T), ('dwFlags', DWORD), ('dwLockCount', DWORD), ('dwResvd', DWORD), ('th32ProcessID', DWORD), ('th32HeapID', ULONG_PTR)]