import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class THREADNAME_INFO(Structure):
    _fields_ = [('dwType', DWORD), ('szName', LPVOID), ('dwThreadID', DWORD), ('dwFlags', DWORD)]