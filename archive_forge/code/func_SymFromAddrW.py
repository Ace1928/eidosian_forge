from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymFromAddrW(hProcess, Address):
    _SymFromAddr = windll.dbghelp.SymFromAddrW
    _SymFromAddr.argtypes = [HANDLE, DWORD64, PDWORD64, PSYM_INFOW]
    _SymFromAddr.restype = bool
    _SymFromAddr.errcheck = RaiseIfZero
    SymInfo = SYM_INFOW()
    SymInfo.SizeOfStruct = 88
    SymInfo.MaxNameLen = MAX_SYM_NAME
    Displacement = DWORD64(0)
    _SymFromAddr(hProcess, Address, byref(Displacement), byref(SymInfo))
    return (Displacement.value, SymInfo)