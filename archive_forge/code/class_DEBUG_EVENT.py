import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class DEBUG_EVENT(Structure):
    _fields_ = [('dwDebugEventCode', DWORD), ('dwProcessId', DWORD), ('dwThreadId', DWORD), ('u', _DEBUG_EVENT_UNION_)]