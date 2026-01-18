from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def CopySid(pSourceSid):
    _CopySid = windll.advapi32.CopySid
    _CopySid.argtypes = [DWORD, PVOID, PSID]
    _CopySid.restype = bool
    _CopySid.errcheck = RaiseIfZero
    nDestinationSidLength = GetLengthSid(pSourceSid)
    DestinationSid = ctypes.create_string_buffer('', nDestinationSidLength)
    pDestinationSid = ctypes.cast(ctypes.pointer(DestinationSid), PVOID)
    _CopySid(nDestinationSidLength, pDestinationSid, pSourceSid)
    return ctypes.cast(pDestinationSid, PSID)