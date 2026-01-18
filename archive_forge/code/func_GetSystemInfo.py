from winappdbg.win32.defines import *
def GetSystemInfo():
    _GetSystemInfo = windll.kernel32.GetSystemInfo
    _GetSystemInfo.argtypes = [LPSYSTEM_INFO]
    _GetSystemInfo.restype = None
    sysinfo = SYSTEM_INFO()
    _GetSystemInfo(byref(sysinfo))
    return sysinfo