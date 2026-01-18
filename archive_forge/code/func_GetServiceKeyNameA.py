from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetServiceKeyNameA(hSCManager, lpDisplayName):
    _GetServiceKeyNameA = windll.advapi32.GetServiceKeyNameA
    _GetServiceKeyNameA.argtypes = [SC_HANDLE, LPSTR, LPSTR, LPDWORD]
    _GetServiceKeyNameA.restype = bool
    cchBuffer = DWORD(0)
    _GetServiceKeyNameA(hSCManager, lpDisplayName, None, byref(cchBuffer))
    if cchBuffer.value == 0:
        raise ctypes.WinError()
    lpServiceName = ctypes.create_string_buffer(cchBuffer.value + 1)
    cchBuffer.value = sizeof(lpServiceName)
    success = _GetServiceKeyNameA(hSCManager, lpDisplayName, lpServiceName, byref(cchBuffer))
    if not success:
        raise ctypes.WinError()
    return lpServiceName.value