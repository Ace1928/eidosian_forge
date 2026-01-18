from winappdbg.win32.defines import *
def GetVersionExW():
    _GetVersionExW = windll.kernel32.GetVersionExW
    _GetVersionExW.argtypes = [POINTER(OSVERSIONINFOEXW)]
    _GetVersionExW.restype = bool
    _GetVersionExW.errcheck = RaiseIfZero
    osi = OSVERSIONINFOEXW()
    osi.dwOSVersionInfoSize = sizeof(osi)
    try:
        _GetVersionExW(byref(osi))
    except WindowsError:
        osi = OSVERSIONINFOW()
        osi.dwOSVersionInfoSize = sizeof(osi)
        _GetVersionExW.argtypes = [POINTER(OSVERSIONINFOW)]
        _GetVersionExW(byref(osi))
    return osi