from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class _WAITCHAIN_NODE_INFO_STRUCT_2(Structure):
    _fields_ = [('ProcessId', DWORD), ('ThreadId', DWORD), ('WaitTime', DWORD), ('ContextSwitches', DWORD)]