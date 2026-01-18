from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def ConvertStringSidToSidA(StringSid):
    _ConvertStringSidToSidA = windll.advapi32.ConvertStringSidToSidA
    _ConvertStringSidToSidA.argtypes = [LPSTR, PVOID]
    _ConvertStringSidToSidA.restype = bool
    _ConvertStringSidToSidA.errcheck = RaiseIfZero
    Sid = PVOID()
    _ConvertStringSidToSidA(StringSid, ctypes.pointer(Sid))
    return Sid.value