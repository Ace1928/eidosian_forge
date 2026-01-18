from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetSearchPathA(hProcess, SearchPath=None):
    _SymSetSearchPath = windll.dbghelp.SymSetSearchPath
    _SymSetSearchPath.argtypes = [HANDLE, LPSTR]
    _SymSetSearchPath.restype = bool
    _SymSetSearchPath.errcheck = RaiseIfZero
    if not SearchPath:
        SearchPath = None
    _SymSetSearchPath(hProcess, SearchPath)