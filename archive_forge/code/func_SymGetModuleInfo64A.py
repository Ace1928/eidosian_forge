from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetModuleInfo64A(hProcess, dwAddr):
    _SymGetModuleInfo64 = windll.dbghelp.SymGetModuleInfo64
    _SymGetModuleInfo64.argtypes = [HANDLE, DWORD64, PIMAGEHLP_MODULE64]
    _SymGetModuleInfo64.restype = bool
    _SymGetModuleInfo64.errcheck = RaiseIfZero
    ModuleInfo = IMAGEHLP_MODULE64()
    ModuleInfo.SizeOfStruct = sizeof(ModuleInfo)
    _SymGetModuleInfo64(hProcess, dwAddr, byref(ModuleInfo))
    return ModuleInfo