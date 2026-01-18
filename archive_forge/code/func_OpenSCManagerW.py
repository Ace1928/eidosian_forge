from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenSCManagerW(lpMachineName=None, lpDatabaseName=None, dwDesiredAccess=SC_MANAGER_ALL_ACCESS):
    _OpenSCManagerW = windll.advapi32.OpenSCManagerW
    _OpenSCManagerW.argtypes = [LPWSTR, LPWSTR, DWORD]
    _OpenSCManagerW.restype = SC_HANDLE
    _OpenSCManagerW.errcheck = RaiseIfZero
    hSCObject = _OpenSCManagerA(lpMachineName, lpDatabaseName, dwDesiredAccess)
    return ServiceControlManagerHandle(hSCObject)