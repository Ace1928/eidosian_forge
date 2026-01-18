from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateSymbolsW(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext=None):
    _SymEnumerateSymbolsW = windll.dbghelp.SymEnumerateSymbolsW
    _SymEnumerateSymbolsW.argtypes = [HANDLE, ULONG, PSYM_ENUMSYMBOLS_CALLBACKW, PVOID]
    _SymEnumerateSymbolsW.restype = bool
    _SymEnumerateSymbolsW.errcheck = RaiseIfZero
    EnumSymbolsCallback = PSYM_ENUMSYMBOLS_CALLBACKW(EnumSymbolsCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateSymbolsW(hProcess, BaseOfDll, EnumSymbolsCallback, UserContext)