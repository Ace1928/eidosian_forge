from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def MakeSureDirectoryPathExistsA(DirPath):
    _MakeSureDirectoryPathExists = windll.dbghelp.MakeSureDirectoryPathExists
    _MakeSureDirectoryPathExists.argtypes = [LPSTR]
    _MakeSureDirectoryPathExists.restype = bool
    _MakeSureDirectoryPathExists.errcheck = RaiseIfZero
    return _MakeSureDirectoryPathExists(DirPath)