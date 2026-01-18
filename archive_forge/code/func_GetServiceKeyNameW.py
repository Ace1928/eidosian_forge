from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetServiceKeyNameW(hSCManager, lpDisplayName):
    _GetServiceKeyNameW = windll.advapi32.GetServiceKeyNameW
    _GetServiceKeyNameW.argtypes = [SC_HANDLE, LPWSTR, LPWSTR, LPDWORD]
    _GetServiceKeyNameW.restype = bool
    cchBuffer = DWORD(0)
    _GetServiceKeyNameW(hSCManager, lpDisplayName, None, byref(cchBuffer))
    if cchBuffer.value == 0:
        raise ctypes.WinError()
    lpServiceName = ctypes.create_unicode_buffer(cchBuffer.value + 2)
    cchBuffer.value = sizeof(lpServiceName)
    success = _GetServiceKeyNameW(hSCManager, lpDisplayName, lpServiceName, byref(cchBuffer))
    if not success:
        raise ctypes.WinError()
    return lpServiceName.value