from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetOptions():
    _SymGetOptions = windll.dbghelp.SymGetOptions
    _SymGetOptions.argtypes = []
    _SymGetOptions.restype = DWORD
    return _SymGetOptions()