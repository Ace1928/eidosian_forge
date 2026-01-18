from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenServiceW(hSCManager, lpServiceName, dwDesiredAccess=SERVICE_ALL_ACCESS):
    _OpenServiceW = windll.advapi32.OpenServiceW
    _OpenServiceW.argtypes = [SC_HANDLE, LPWSTR, DWORD]
    _OpenServiceW.restype = SC_HANDLE
    _OpenServiceW.errcheck = RaiseIfZero
    return ServiceHandle(_OpenServiceW(hSCManager, lpServiceName, dwDesiredAccess))