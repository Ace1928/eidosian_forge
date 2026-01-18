from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetServiceDisplayNameA(hSCManager, lpServiceName):
    _GetServiceDisplayNameA = windll.advapi32.GetServiceDisplayNameA
    _GetServiceDisplayNameA.argtypes = [SC_HANDLE, LPSTR, LPSTR, LPDWORD]
    _GetServiceDisplayNameA.restype = bool
    cchBuffer = DWORD(0)
    _GetServiceDisplayNameA(hSCManager, lpServiceName, None, byref(cchBuffer))
    if cchBuffer.value == 0:
        raise ctypes.WinError()
    lpDisplayName = ctypes.create_string_buffer(cchBuffer.value + 1)
    cchBuffer.value = sizeof(lpDisplayName)
    success = _GetServiceDisplayNameA(hSCManager, lpServiceName, lpDisplayName, byref(cchBuffer))
    if not success:
        raise ctypes.WinError()
    return lpDisplayName.value