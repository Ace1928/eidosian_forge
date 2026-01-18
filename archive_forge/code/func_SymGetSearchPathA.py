from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetSearchPathA(hProcess):
    _SymGetSearchPath = windll.dbghelp.SymGetSearchPath
    _SymGetSearchPath.argtypes = [HANDLE, LPSTR, DWORD]
    _SymGetSearchPath.restype = bool
    _SymGetSearchPath.errcheck = RaiseIfZero
    SearchPathLength = MAX_PATH
    SearchPath = ctypes.create_string_buffer('', SearchPathLength)
    _SymGetSearchPath(hProcess, SearchPath, SearchPathLength)
    return SearchPath.value