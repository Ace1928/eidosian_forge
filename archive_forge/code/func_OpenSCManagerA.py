from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenSCManagerA(lpMachineName=None, lpDatabaseName=None, dwDesiredAccess=SC_MANAGER_ALL_ACCESS):
    _OpenSCManagerA = windll.advapi32.OpenSCManagerA
    _OpenSCManagerA.argtypes = [LPSTR, LPSTR, DWORD]
    _OpenSCManagerA.restype = SC_HANDLE
    _OpenSCManagerA.errcheck = RaiseIfZero
    hSCObject = _OpenSCManagerA(lpMachineName, lpDatabaseName, dwDesiredAccess)
    return ServiceControlManagerHandle(hSCObject)