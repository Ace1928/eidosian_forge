import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class MODULEENTRY32(Structure):
    _fields_ = [('dwSize', DWORD), ('th32ModuleID', DWORD), ('th32ProcessID', DWORD), ('GlblcntUsage', DWORD), ('ProccntUsage', DWORD), ('modBaseAddr', LPVOID), ('modBaseSize', DWORD), ('hModule', HMODULE), ('szModule', TCHAR * (MAX_MODULE_NAME32 + 1)), ('szExePath', TCHAR * MAX_PATH)]