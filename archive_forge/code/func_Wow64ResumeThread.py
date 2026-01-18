from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
def Wow64ResumeThread(hThread):
    _Wow64ResumeThread = windll.kernel32.Wow64ResumeThread
    _Wow64ResumeThread.argtypes = [HANDLE]
    _Wow64ResumeThread.restype = DWORD
    previousCount = _Wow64ResumeThread(hThread)
    if previousCount == DWORD(-1).value:
        raise ctypes.WinError()
    return previousCount