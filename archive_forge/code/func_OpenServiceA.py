from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenServiceA(hSCManager, lpServiceName, dwDesiredAccess=SERVICE_ALL_ACCESS):
    _OpenServiceA = windll.advapi32.OpenServiceA
    _OpenServiceA.argtypes = [SC_HANDLE, LPSTR, DWORD]
    _OpenServiceA.restype = SC_HANDLE
    _OpenServiceA.errcheck = RaiseIfZero
    return ServiceHandle(_OpenServiceA(hSCManager, lpServiceName, dwDesiredAccess))