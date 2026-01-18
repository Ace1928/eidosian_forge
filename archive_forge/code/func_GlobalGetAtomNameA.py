import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalGetAtomNameA(nAtom):
    _GlobalGetAtomNameA = windll.kernel32.GlobalGetAtomNameA
    _GlobalGetAtomNameA.argtypes = [ATOM, LPSTR, ctypes.c_int]
    _GlobalGetAtomNameA.restype = UINT
    _GlobalGetAtomNameA.errcheck = RaiseIfZero
    nSize = 64
    while 1:
        lpBuffer = ctypes.create_string_buffer('', nSize)
        nCopied = _GlobalGetAtomNameA(nAtom, lpBuffer, nSize)
        if nCopied < nSize - 1:
            break
        nSize = nSize + 64
    return lpBuffer.value