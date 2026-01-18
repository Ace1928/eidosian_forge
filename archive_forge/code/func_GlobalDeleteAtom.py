import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalDeleteAtom(nAtom):
    _GlobalDeleteAtom = windll.kernel32.GlobalDeleteAtom
    _GlobalDeleteAtom.argtypes
    _GlobalDeleteAtom.restype
    SetLastError(ERROR_SUCCESS)
    _GlobalDeleteAtom(nAtom)
    error = GetLastError()
    if error != ERROR_SUCCESS:
        raise ctypes.WinError(error)