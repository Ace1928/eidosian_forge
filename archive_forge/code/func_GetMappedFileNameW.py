from winappdbg.win32.defines import *
def GetMappedFileNameW(hProcess, lpv):
    _GetMappedFileNameW = ctypes.windll.psapi.GetMappedFileNameW
    _GetMappedFileNameW.argtypes = [HANDLE, LPVOID, LPWSTR, DWORD]
    _GetMappedFileNameW.restype = DWORD
    nSize = MAX_PATH
    while 1:
        lpFilename = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = _GetMappedFileNameW(hProcess, lpv, lpFilename, nSize)
        if nCopied == 0:
            raise ctypes.WinError()
        if nCopied < nSize - 1:
            break
        nSize = nSize + MAX_PATH
    return lpFilename.value