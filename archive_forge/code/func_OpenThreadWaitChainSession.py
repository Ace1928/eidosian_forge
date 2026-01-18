from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def OpenThreadWaitChainSession(Flags=0, callback=None):
    _OpenThreadWaitChainSession = windll.advapi32.OpenThreadWaitChainSession
    _OpenThreadWaitChainSession.argtypes = [DWORD, PVOID]
    _OpenThreadWaitChainSession.restype = HWCT
    _OpenThreadWaitChainSession.errcheck = RaiseIfZero
    if callback is not None:
        callback = PWAITCHAINCALLBACK(callback)
    aHandle = _OpenThreadWaitChainSession(Flags, callback)
    return ThreadWaitChainSessionHandle(aHandle)