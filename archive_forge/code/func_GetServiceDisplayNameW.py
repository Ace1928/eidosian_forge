from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetServiceDisplayNameW(hSCManager, lpServiceName):
    _GetServiceDisplayNameW = windll.advapi32.GetServiceDisplayNameW
    _GetServiceDisplayNameW.argtypes = [SC_HANDLE, LPWSTR, LPWSTR, LPDWORD]
    _GetServiceDisplayNameW.restype = bool
    cchBuffer = DWORD(0)
    _GetServiceDisplayNameW(hSCManager, lpServiceName, None, byref(cchBuffer))
    if cchBuffer.value == 0:
        raise ctypes.WinError()
    lpDisplayName = ctypes.create_unicode_buffer(cchBuffer.value + 2)
    cchBuffer.value = sizeof(lpDisplayName)
    success = _GetServiceDisplayNameW(hSCManager, lpServiceName, lpDisplayName, byref(cchBuffer))
    if not success:
        raise ctypes.WinError()
    return lpDisplayName.value