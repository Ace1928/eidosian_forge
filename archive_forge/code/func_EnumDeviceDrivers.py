from winappdbg.win32.defines import *
def EnumDeviceDrivers():
    _EnumDeviceDrivers = windll.psapi.EnumDeviceDrivers
    _EnumDeviceDrivers.argtypes = [LPVOID, DWORD, LPDWORD]
    _EnumDeviceDrivers.restype = bool
    _EnumDeviceDrivers.errcheck = RaiseIfZero
    size = 4096
    lpcbNeeded = DWORD(size)
    unit = sizeof(LPVOID)
    while 1:
        lpImageBase = (LPVOID * (size // unit))()
        _EnumDeviceDrivers(byref(lpImageBase), lpcbNeeded, byref(lpcbNeeded))
        needed = lpcbNeeded.value
        if needed <= size:
            break
        size = needed
    return [lpImageBase[index] for index in compat.xrange(0, needed // unit)]