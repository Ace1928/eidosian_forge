from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def EnumServicesStatusExW(hSCManager, InfoLevel=SC_ENUM_PROCESS_INFO, dwServiceType=SERVICE_DRIVER | SERVICE_WIN32, dwServiceState=SERVICE_STATE_ALL, pszGroupName=None):
    _EnumServicesStatusExW = windll.advapi32.EnumServicesStatusExW
    _EnumServicesStatusExW.argtypes = [SC_HANDLE, SC_ENUM_TYPE, DWORD, DWORD, LPVOID, DWORD, LPDWORD, LPDWORD, LPDWORD, LPWSTR]
    _EnumServicesStatusExW.restype = bool
    if InfoLevel != SC_ENUM_PROCESS_INFO:
        raise NotImplementedError()
    cbBytesNeeded = DWORD(0)
    ServicesReturned = DWORD(0)
    ResumeHandle = DWORD(0)
    _EnumServicesStatusExW(hSCManager, InfoLevel, dwServiceType, dwServiceState, None, 0, byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle), pszGroupName)
    Services = []
    success = False
    while GetLastError() == ERROR_MORE_DATA:
        if cbBytesNeeded.value < sizeof(ENUM_SERVICE_STATUS_PROCESSW):
            break
        ServicesBuffer = ctypes.create_string_buffer('', cbBytesNeeded.value)
        success = _EnumServicesStatusExW(hSCManager, InfoLevel, dwServiceType, dwServiceState, byref(ServicesBuffer), sizeof(ServicesBuffer), byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle), pszGroupName)
        if sizeof(ServicesBuffer) < sizeof(ENUM_SERVICE_STATUS_PROCESSW) * ServicesReturned.value:
            raise ctypes.WinError()
        lpServicesArray = ctypes.cast(ctypes.cast(ctypes.pointer(ServicesBuffer), ctypes.c_void_p), LPENUM_SERVICE_STATUS_PROCESSW)
        for index in compat.xrange(0, ServicesReturned.value):
            Services.append(ServiceStatusProcessEntry(lpServicesArray[index]))
        if success:
            break
    if not success:
        raise ctypes.WinError()
    return Services