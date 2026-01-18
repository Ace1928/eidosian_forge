from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymCleanup(hProcess):
    _SymCleanup = windll.dbghelp.SymCleanup
    _SymCleanup.argtypes = [HANDLE]
    _SymCleanup.restype = bool
    _SymCleanup.errcheck = RaiseIfZero
    _SymCleanup(hProcess)