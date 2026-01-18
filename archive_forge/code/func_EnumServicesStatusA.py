from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def EnumServicesStatusA(hSCManager, dwServiceType=SERVICE_DRIVER | SERVICE_WIN32, dwServiceState=SERVICE_STATE_ALL):
    _EnumServicesStatusA = windll.advapi32.EnumServicesStatusA
    _EnumServicesStatusA.argtypes = [SC_HANDLE, DWORD, DWORD, LPVOID, DWORD, LPDWORD, LPDWORD, LPDWORD]
    _EnumServicesStatusA.restype = bool
    cbBytesNeeded = DWORD(0)
    ServicesReturned = DWORD(0)
    ResumeHandle = DWORD(0)
    _EnumServicesStatusA(hSCManager, dwServiceType, dwServiceState, None, 0, byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle))
    Services = []
    success = False
    while GetLastError() == ERROR_MORE_DATA:
        if cbBytesNeeded.value < sizeof(ENUM_SERVICE_STATUSA):
            break
        ServicesBuffer = ctypes.create_string_buffer('', cbBytesNeeded.value)
        success = _EnumServicesStatusA(hSCManager, dwServiceType, dwServiceState, byref(ServicesBuffer), sizeof(ServicesBuffer), byref(cbBytesNeeded), byref(ServicesReturned), byref(ResumeHandle))
        if sizeof(ServicesBuffer) < sizeof(ENUM_SERVICE_STATUSA) * ServicesReturned.value:
            raise ctypes.WinError()
        lpServicesArray = ctypes.cast(ctypes.cast(ctypes.pointer(ServicesBuffer), ctypes.c_void_p), LPENUM_SERVICE_STATUSA)
        for index in compat.xrange(0, ServicesReturned.value):
            Services.append(ServiceStatusEntry(lpServicesArray[index]))
        if success:
            break
    if not success:
        raise ctypes.WinError()
    return Services