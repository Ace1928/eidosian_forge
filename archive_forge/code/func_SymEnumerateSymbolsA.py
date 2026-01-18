from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateSymbolsA(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext=None):
    _SymEnumerateSymbols = windll.dbghelp.SymEnumerateSymbols
    _SymEnumerateSymbols.argtypes = [HANDLE, ULONG, PSYM_ENUMSYMBOLS_CALLBACK, PVOID]
    _SymEnumerateSymbols.restype = bool
    _SymEnumerateSymbols.errcheck = RaiseIfZero
    EnumSymbolsCallback = PSYM_ENUMSYMBOLS_CALLBACK(EnumSymbolsCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateSymbols(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)