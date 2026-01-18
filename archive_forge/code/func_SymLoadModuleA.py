from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymLoadModuleA(hProcess, hFile=None, ImageName=None, ModuleName=None, BaseOfDll=None, SizeOfDll=None):
    _SymLoadModule = windll.dbghelp.SymLoadModule
    _SymLoadModule.argtypes = [HANDLE, HANDLE, LPSTR, LPSTR, DWORD, DWORD]
    _SymLoadModule.restype = DWORD
    if not ImageName:
        ImageName = None
    if not ModuleName:
        ModuleName = None
    if not BaseOfDll:
        BaseOfDll = 0
    if not SizeOfDll:
        SizeOfDll = 0
    SetLastError(ERROR_SUCCESS)
    lpBaseAddress = _SymLoadModule(hProcess, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll)
    if lpBaseAddress == NULL:
        dwErrorCode = GetLastError()
        if dwErrorCode != ERROR_SUCCESS:
            raise ctypes.WinError(dwErrorCode)
    return lpBaseAddress