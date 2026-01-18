from winappdbg.win32.defines import *
def GetNativeSystemInfo():
    _GetNativeSystemInfo = windll.kernel32.GetNativeSystemInfo
    _GetNativeSystemInfo.argtypes = [LPSYSTEM_INFO]
    _GetNativeSystemInfo.restype = None
    sysinfo = SYSTEM_INFO()
    _GetNativeSystemInfo(byref(sysinfo))
    return sysinfo