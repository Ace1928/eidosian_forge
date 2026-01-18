from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
def SetThreadContext(hThread, lpContext):
    _SetThreadContext = windll.kernel32.SetThreadContext
    _SetThreadContext.argtypes = [HANDLE, LPCONTEXT]
    _SetThreadContext.restype = bool
    _SetThreadContext.errcheck = RaiseIfZero
    if isinstance(lpContext, dict):
        lpContext = CONTEXT.from_dict(lpContext)
    _SetThreadContext(hThread, byref(lpContext))