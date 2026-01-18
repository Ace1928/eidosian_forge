import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class CREATE_PROCESS_DEBUG_INFO(Structure):
    _fields_ = [('hFile', HANDLE), ('hProcess', HANDLE), ('hThread', HANDLE), ('lpBaseOfImage', LPVOID), ('dwDebugInfoFileOffset', DWORD), ('nDebugInfoSize', DWORD), ('lpThreadLocalBase', LPVOID), ('lpStartAddress', LPVOID), ('lpImageName', LPVOID), ('fUnicode', WORD)]