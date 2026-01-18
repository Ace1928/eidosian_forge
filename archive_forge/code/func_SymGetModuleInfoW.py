from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetModuleInfoW(hProcess, dwAddr):
    _SymGetModuleInfoW = windll.dbghelp.SymGetModuleInfoW
    _SymGetModuleInfoW.argtypes = [HANDLE, DWORD, PIMAGEHLP_MODULEW]
    _SymGetModuleInfoW.restype = bool
    _SymGetModuleInfoW.errcheck = RaiseIfZero
    ModuleInfo = IMAGEHLP_MODULEW()
    ModuleInfo.SizeOfStruct = sizeof(ModuleInfo)
    _SymGetModuleInfoW(hProcess, dwAddr, byref(ModuleInfo))
    return ModuleInfo