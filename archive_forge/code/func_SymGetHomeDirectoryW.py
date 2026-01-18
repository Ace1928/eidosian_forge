from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetHomeDirectoryW(type):
    _SymGetHomeDirectoryW = windll.dbghelp.SymGetHomeDirectoryW
    _SymGetHomeDirectoryW.argtypes = [DWORD, LPWSTR, SIZE_T]
    _SymGetHomeDirectoryW.restype = LPWSTR
    _SymGetHomeDirectoryW.errcheck = RaiseIfZero
    size = MAX_PATH
    dir = ctypes.create_unicode_buffer(u'', size)
    _SymGetHomeDirectoryW(type, dir, size)
    return dir.value