from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def ConvertSidToStringSidW(Sid):
    _ConvertSidToStringSidW = windll.advapi32.ConvertSidToStringSidW
    _ConvertSidToStringSidW.argtypes = [PSID, LPWSTR]
    _ConvertSidToStringSidW.restype = bool
    _ConvertSidToStringSidW.errcheck = RaiseIfZero
    pStringSid = LPWSTR()
    _ConvertSidToStringSidW(Sid, byref(pStringSid))
    try:
        StringSid = pStringSid.value
    finally:
        LocalFree(pStringSid)
    return StringSid