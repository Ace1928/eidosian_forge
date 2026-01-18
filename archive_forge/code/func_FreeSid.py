from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def FreeSid(pSid):
    _FreeSid = windll.advapi32.FreeSid
    _FreeSid.argtypes = [PSID]
    _FreeSid.restype = PSID
    _FreeSid.errcheck = RaiseIfNotZero
    _FreeSid(pSid)