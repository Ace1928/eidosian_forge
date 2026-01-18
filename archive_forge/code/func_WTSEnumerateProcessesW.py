from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def WTSEnumerateProcessesW(hServer=WTS_CURRENT_SERVER_HANDLE):
    _WTSEnumerateProcessesW = windll.wtsapi32.WTSEnumerateProcessesW
    _WTSEnumerateProcessesW.argtypes = [HANDLE, DWORD, DWORD, POINTER(PWTS_PROCESS_INFOW), PDWORD]
    _WTSEnumerateProcessesW.restype = bool
    _WTSEnumerateProcessesW.errcheck = RaiseIfZero
    pProcessInfo = PWTS_PROCESS_INFOW()
    Count = DWORD(0)
    _WTSEnumerateProcessesW(hServer, 0, 1, byref(pProcessInfo), byref(Count))
    return (pProcessInfo, Count.value)