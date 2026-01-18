from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetHomeDirectoryA(hProcess, dir=None):
    _SymSetHomeDirectoryA = windll.dbghelp.SymSetHomeDirectoryA
    _SymSetHomeDirectoryA.argtypes = [HANDLE, LPSTR]
    _SymSetHomeDirectoryA.restype = LPSTR
    _SymSetHomeDirectoryA.errcheck = RaiseIfZero
    if not dir:
        dir = None
    _SymSetHomeDirectoryA(hProcess, dir)
    return dir