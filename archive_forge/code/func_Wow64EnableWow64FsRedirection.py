import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Wow64EnableWow64FsRedirection(Wow64FsEnableRedirection):
    """
    This function may not work reliably when there are nested calls. Therefore,
    this function has been replaced by the L{Wow64DisableWow64FsRedirection}
    and L{Wow64RevertWow64FsRedirection} functions.

    @see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/aa365744(v=vs.85).aspx}
    """
    _Wow64EnableWow64FsRedirection = windll.kernel32.Wow64EnableWow64FsRedirection
    _Wow64EnableWow64FsRedirection.argtypes = [BOOLEAN]
    _Wow64EnableWow64FsRedirection.restype = BOOLEAN
    _Wow64EnableWow64FsRedirection.errcheck = RaiseIfZero