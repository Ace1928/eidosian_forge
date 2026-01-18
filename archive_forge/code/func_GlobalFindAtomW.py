import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalFindAtomW(lpString):
    _GlobalFindAtomW = windll.kernel32.GlobalFindAtomW
    _GlobalFindAtomW.argtypes = [LPWSTR]
    _GlobalFindAtomW.restype = ATOM
    _GlobalFindAtomW.errcheck = RaiseIfZero
    return _GlobalFindAtomW(lpString)