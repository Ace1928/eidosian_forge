from winappdbg.win32.defines import *
def GetModuleFileNameExW(hProcess, hModule=None):
    _GetModuleFileNameExW = ctypes.windll.psapi.GetModuleFileNameExW
    _GetModuleFileNameExW.argtypes = [HANDLE, HMODULE, LPWSTR, DWORD]
    _GetModuleFileNameExW.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = _GetModuleFileNameExW(hProcess, hModule, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value