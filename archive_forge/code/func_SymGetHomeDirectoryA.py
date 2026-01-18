from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetHomeDirectoryA(type):
    _SymGetHomeDirectoryA = windll.dbghelp.SymGetHomeDirectoryA
    _SymGetHomeDirectoryA.argtypes = [DWORD, LPSTR, SIZE_T]
    _SymGetHomeDirectoryA.restype = LPSTR
    _SymGetHomeDirectoryA.errcheck = RaiseIfZero
    size = MAX_PATH
    dir = ctypes.create_string_buffer('', size)
    _SymGetHomeDirectoryA(type, dir, size)
    return dir.value