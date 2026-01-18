from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymLoadModule64A(hProcess, hFile=None, ImageName=None, ModuleName=None, BaseOfDll=None, SizeOfDll=None):
    _SymLoadModule64 = windll.dbghelp.SymLoadModule64
    _SymLoadModule64.argtypes = [HANDLE, HANDLE, LPSTR, LPSTR, DWORD64, DWORD]
    _SymLoadModule64.restype = DWORD64
    if not ImageName:
        ImageName = None
    if not ModuleName:
        ModuleName = None
    if not BaseOfDll:
        BaseOfDll = 0
    if not SizeOfDll:
        SizeOfDll = 0
    SetLastError(ERROR_SUCCESS)
    lpBaseAddress = _SymLoadModule64(hProcess, hFile, ImageName, ModuleName, BaseOfDll, SizeOfDll)
    if lpBaseAddress == NULL:
        dwErrorCode = GetLastError()
        if dwErrorCode != ERROR_SUCCESS:
            raise ctypes.WinError(dwErrorCode)
    return lpBaseAddress