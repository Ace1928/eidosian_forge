from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetLengthSid(pSid):
    _GetLengthSid = windll.advapi32.GetLengthSid
    _GetLengthSid.argtypes = [PSID]
    _GetLengthSid.restype = DWORD
    return _GetLengthSid(pSid)