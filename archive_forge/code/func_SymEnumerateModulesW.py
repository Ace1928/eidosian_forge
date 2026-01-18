from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateModulesW(hProcess, EnumModulesCallback, UserContext=None):
    _SymEnumerateModulesW = windll.dbghelp.SymEnumerateModulesW
    _SymEnumerateModulesW.argtypes = [HANDLE, PSYM_ENUMMODULES_CALLBACKW, PVOID]
    _SymEnumerateModulesW.restype = bool
    _SymEnumerateModulesW.errcheck = RaiseIfZero
    EnumModulesCallback = PSYM_ENUMMODULES_CALLBACKW(EnumModulesCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateModulesW(hProcess, EnumModulesCallback, UserContext)