from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def EnumServicesStatusW(hSCManager, dwServiceType=SERVICE_DRIVER | SERVICE_WIN32, dwServiceState=SERVICE_STATE_ALL):
    _EnumServicesStatusW = windll.advapi32.EnumServicesStatusW
    _EnumServicesStatusW.argtypes = [SC_HANDLE, DWORD, DWORD, LPVOID, DWORD, LPDWORD, LPDWORD, LPDWORD]
    _EnumServicesStatusW.restype = bool
    cbBytesNeeded = DWORD(0)
    ServicesReturned = DWORD(0)
    ResumeHandle = DWORD(0)
    _EnumServicesStatusW(hSCManager, dwServiceType, dwServiceState, None, 0, byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle))
    Services = []
    success = False
    while GetLastError() == ERROR_MORE_DATA:
        if cbBytesNeeded.value < sizeof(ENUM_SERVICE_STATUSW):
            break
        ServicesBuffer = ctypes.create_string_buffer('', cbBytesNeeded.value)
        success = _EnumServicesStatusW(hSCManager, dwServiceType, dwServiceState, byref(ServicesBuffer), sizeof(ServicesBuffer), byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle))
        if sizeof(ServicesBuffer) < sizeof(ENUM_SERVICE_STATUSW) * ServicesReturned.value:
            raise ctypes.WinError()
        lpServicesArray = ctypes.cast(ctypes.cast(ctypes.pointer(ServicesBuffer), ctypes.c_void_p), LPENUM_SERVICE_STATUSW)
        for index in compat.xrange(0, ServicesReturned.value):
            Services.append(ServiceStatusEntry(lpServicesArray[index]))
        if success:
            break
    if not success:
        raise ctypes.WinError()
    return Services