import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalAddAtomW(lpString):
    _GlobalAddAtomW = windll.kernel32.GlobalAddAtomW
    _GlobalAddAtomW.argtypes = [LPWSTR]
    _GlobalAddAtomW.restype = ATOM
    _GlobalAddAtomW.errcheck = RaiseIfZero
    return _GlobalAddAtomW(lpString)