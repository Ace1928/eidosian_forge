from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def CreateProcessWithLogonW(lpUsername=None, lpDomain=None, lpPassword=None, dwLogonFlags=0, lpApplicationName=None, lpCommandLine=None, dwCreationFlags=0, lpEnvironment=None, lpCurrentDirectory=None, lpStartupInfo=None):
    _CreateProcessWithLogonW = windll.advapi32.CreateProcessWithLogonW
    _CreateProcessWithLogonW.argtypes = [LPWSTR, LPWSTR, LPWSTR, DWORD, LPWSTR, LPWSTR, DWORD, LPVOID, LPWSTR, LPVOID, LPPROCESS_INFORMATION]
    _CreateProcessWithLogonW.restype = bool
    _CreateProcessWithLogonW.errcheck = RaiseIfZero
    if not lpUsername:
        lpUsername = None
    if not lpDomain:
        lpDomain = None
    if not lpPassword:
        lpPassword = None
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
    if not lpStartupInfo:
        lpStartupInfo = STARTUPINFOW()
        lpStartupInfo.cb = sizeof(STARTUPINFOW)
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
    _CreateProcessWithLogonW(lpUsername, lpDomain, lpPassword, dwLogonFlags, lpApplicationName, lpCommandLine, dwCreationFlags, lpEnvironment, lpCurrentDirectory, byref(lpStartupInfo), byref(lpProcessInformation))
    return ProcessInformation(lpProcessInformation)