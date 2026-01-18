from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def ConvertSidToStringSidA(Sid):
    _ConvertSidToStringSidA = windll.advapi32.ConvertSidToStringSidA
    _ConvertSidToStringSidA.argtypes = [PSID, LPSTR]
    _ConvertSidToStringSidA.restype = bool
    _ConvertSidToStringSidA.errcheck = RaiseIfZero
    pStringSid = LPSTR()
    _ConvertSidToStringSidA(Sid, byref(pStringSid))
    try:
        StringSid = pStringSid.value
    finally:
        LocalFree(pStringSid)
    return StringSid