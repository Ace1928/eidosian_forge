from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateSymbols64A(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext=None):
    _SymEnumerateSymbols64 = windll.dbghelp.SymEnumerateSymbols64
    _SymEnumerateSymbols64.argtypes = [HANDLE, ULONG64, PSYM_ENUMSYMBOLS_CALLBACK64, PVOID]
    _SymEnumerateSymbols64.restype = bool
    _SymEnumerateSymbols64.errcheck = RaiseIfZero
    EnumSymbolsCallback = PSYM_ENUMSYMBOLS_CALLBACK64(EnumSymbolsCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateSymbols64(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)