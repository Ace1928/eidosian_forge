from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetSearchPathW(hProcess, SearchPath=None):
    _SymSetSearchPathW = windll.dbghelp.SymSetSearchPathW
    _SymSetSearchPathW.argtypes = [HANDLE, LPWSTR]
    _SymSetSearchPathW.restype = bool
    _SymSetSearchPathW.errcheck = RaiseIfZero
    if not SearchPath:
        SearchPath = None
    _SymSetSearchPathW(hProcess, SearchPath)