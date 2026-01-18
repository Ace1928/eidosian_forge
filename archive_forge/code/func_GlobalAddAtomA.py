import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalAddAtomA(lpString):
    _GlobalAddAtomA = windll.kernel32.GlobalAddAtomA
    _GlobalAddAtomA.argtypes = [LPSTR]
    _GlobalAddAtomA.restype = ATOM
    _GlobalAddAtomA.errcheck = RaiseIfZero
    return _GlobalAddAtomA(lpString)