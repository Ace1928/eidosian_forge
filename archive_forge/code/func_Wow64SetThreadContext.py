from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
def Wow64SetThreadContext(hThread, lpContext):
    _Wow64SetThreadContext = windll.kernel32.Wow64SetThreadContext
    _Wow64SetThreadContext.argtypes = [HANDLE, PWOW64_CONTEXT]
    _Wow64SetThreadContext.restype = bool
    _Wow64SetThreadContext.errcheck = RaiseIfZero
    if isinstance(lpContext, dict):
        lpContext = WOW64_CONTEXT.from_dict(lpContext)
    _Wow64SetThreadContext(hThread, byref(lpContext))