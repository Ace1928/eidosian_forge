import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class PROCESSENTRY32(Structure):
    _fields_ = [('dwSize', DWORD), ('cntUsage', DWORD), ('th32ProcessID', DWORD), ('th32DefaultHeapID', ULONG_PTR), ('th32ModuleID', DWORD), ('cntThreads', DWORD), ('th32ParentProcessID', DWORD), ('pcPriClassBase', LONG), ('dwFlags', DWORD), ('szExeFile', TCHAR * 260)]