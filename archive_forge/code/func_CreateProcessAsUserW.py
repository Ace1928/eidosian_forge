from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def CreateProcessAsUserW(hToken=None, lpApplicationName=None, lpCommandLine=None, lpProcessAttributes=None, lpThreadAttributes=None, bInheritHandles=False, dwCreationFlags=0, lpEnvironment=None, lpCurrentDirectory=None, lpStartupInfo=None):
    _CreateProcessAsUserW = windll.advapi32.CreateProcessAsUserW
    _CreateProcessAsUserW.argtypes = [HANDLE, LPWSTR, LPWSTR, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES, BOOL, DWORD, LPVOID, LPWSTR, LPVOID, LPPROCESS_INFORMATION]
    _CreateProcessAsUserW.restype = bool
    _CreateProcessAsUserW.errcheck = RaiseIfZero
    if not lpApplicationName:
        lpApplicationName = None
    if not lpCommandLine:
        lpCommandLine = None
    else:
        lpCommandLine = ctypes.create_unicode_buffer(lpCommandLine, max(MAX_PATH, len(lpCommandLine)))
    if not lpEnvironment:
        lpEnvironment = None
    else:
        lpEnvironment = ctypes.create_unicode_buffer(lpEnvironment)
    if not lpCurrentDirectory:
        lpCurrentDirectory = None
    if not lpProcessAttributes:
        lpProcessAttributes = None
    else:
        lpProcessAttributes = byref(lpProcessAttributes)
    if not lpThreadAttributes:
        lpThreadAttributes = None
    else:
        lpThreadAttributes = byref(lpThreadAttributes)
    if not lpStartupInfo:
        lpStartupInfo = STARTUPINFO()
        lpStartupInfo.cb = sizeof(STARTUPINFO)
        lpStartupInfo.lpReserved = 0
        lpStartupInfo.lpDesktop = 0
        lpStartupInfo.lpTitle = 0
        lpStartupInfo.dwFlags = 0
        lpStartupInfo.cbReserved2 = 0
        lpStartupInfo.lpReserved2 = 0
    lpProcessInformation = PROCESS_INFORMATION()
    lpProcessInformation.hProcess = INVALID_HANDLE_VALUE
    lpProcessInformation.hThread = INVALID_HANDLE_VALUE
    lpProcessInformation.dwProcessId = 0
    lpProcessInformation.dwThreadId = 0
    _CreateProcessAsUserW(hToken, lpApplicationName, lpCommandLine, lpProcessAttributes, lpThreadAttributes, bool(bInheritHandles), dwCreationFlags, lpEnvironment, lpCurrentDirectory, byref(lpStartupInfo), byref(lpProcessInformation))
    return ProcessInformation(lpProcessInformation)