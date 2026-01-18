from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateSymbols64W(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext=None):
    _SymEnumerateSymbols64W = windll.dbghelp.SymEnumerateSymbols64W
    _SymEnumerateSymbols64W.argtypes = [HANDLE, ULONG64, PSYM_ENUMSYMBOLS_CALLBACK64W, PVOID]
    _SymEnumerateSymbols64W.restype = bool
    _SymEnumerateSymbols64W.errcheck = RaiseIfZero
    EnumSymbolsCallback = PSYM_ENUMSYMBOLS_CALLBACK64W(EnumSymbolsCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateSymbols64W(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)