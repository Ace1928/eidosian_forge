from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def LookupPrivilegeValueA(lpSystemName, lpName):
    _LookupPrivilegeValueA = windll.advapi32.LookupPrivilegeValueA
    _LookupPrivilegeValueA.argtypes = [LPSTR, LPSTR, PLUID]
    _LookupPrivilegeValueA.restype = bool
    _LookupPrivilegeValueA.errcheck = RaiseIfZero
    lpLuid = LUID()
    if not lpSystemName:
        lpSystemName = None
    _LookupPrivilegeValueA(lpSystemName, lpName, byref(lpLuid))
    return lpLuid