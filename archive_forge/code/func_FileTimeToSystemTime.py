import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FileTimeToSystemTime(lpFileTime):
    _FileTimeToSystemTime = windll.kernel32.FileTimeToSystemTime
    _FileTimeToSystemTime.argtypes = [LPFILETIME, LPSYSTEMTIME]
    _FileTimeToSystemTime.restype = bool
    _FileTimeToSystemTime.errcheck = RaiseIfZero
    if isinstance(lpFileTime, FILETIME):
        FileTime = lpFileTime
    else:
        FileTime = FILETIME()
        FileTime.dwLowDateTime = lpFileTime & 4294967295
        FileTime.dwHighDateTime = lpFileTime >> 32
    SystemTime = SYSTEMTIME()
    _FileTimeToSystemTime(byref(FileTime), byref(SystemTime))
    return SystemTime