import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class THREADENTRY32(Structure):
    _fields_ = [('dwSize', DWORD), ('cntUsage', DWORD), ('th32ThreadID', DWORD), ('th32OwnerProcessID', DWORD), ('tpBasePri', LONG), ('tpDeltaPri', LONG), ('dwFlags', DWORD)]