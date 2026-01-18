from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def CreateServiceW(hSCManager, lpServiceName, lpDisplayName=None, dwDesiredAccess=SERVICE_ALL_ACCESS, dwServiceType=SERVICE_WIN32_OWN_PROCESS, dwStartType=SERVICE_DEMAND_START, dwErrorControl=SERVICE_ERROR_NORMAL, lpBinaryPathName=None, lpLoadOrderGroup=None, lpDependencies=None, lpServiceStartName=None, lpPassword=None):
    _CreateServiceW = windll.advapi32.CreateServiceW
    _CreateServiceW.argtypes = [SC_HANDLE, LPWSTR, LPWSTR, DWORD, DWORD, DWORD, DWORD, LPWSTR, LPWSTR, LPDWORD, LPWSTR, LPWSTR, LPWSTR]
    _CreateServiceW.restype = SC_HANDLE
    _CreateServiceW.errcheck = RaiseIfZero
    dwTagId = DWORD(0)
    hService = _CreateServiceW(hSCManager, lpServiceName, dwDesiredAccess, dwServiceType, dwStartType, dwErrorControl, lpBinaryPathName, lpLoadOrderGroup, byref(dwTagId), lpDependencies, lpServiceStartName, lpPassword)
    return (ServiceHandle(hService), dwTagId.value)