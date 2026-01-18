from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymFromName(hProcess, Name):
    _SymFromNameA = windll.dbghelp.SymFromName
    _SymFromNameA.argtypes = [HANDLE, LPSTR, PSYM_INFO]
    _SymFromNameA.restype = bool
    _SymFromNameA.errcheck = RaiseIfZero
    SymInfo = SYM_INFO()
    SymInfo.SizeOfStruct = 88
    SymInfo.MaxNameLen = MAX_SYM_NAME
    _SymFromNameA(hProcess, Name, byref(SymInfo))
    return SymInfo