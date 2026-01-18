from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymFromNameW(hProcess, Name):
    _SymFromNameW = windll.dbghelp.SymFromNameW
    _SymFromNameW.argtypes = [HANDLE, LPWSTR, PSYM_INFOW]
    _SymFromNameW.restype = bool
    _SymFromNameW.errcheck = RaiseIfZero
    SymInfo = SYM_INFOW()
    SymInfo.SizeOfStruct = 88
    SymInfo.MaxNameLen = MAX_SYM_NAME
    _SymFromNameW(hProcess, Name, byref(SymInfo))
    return SymInfo