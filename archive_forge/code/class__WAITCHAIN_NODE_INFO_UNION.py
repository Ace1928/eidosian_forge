from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class _WAITCHAIN_NODE_INFO_UNION(Union):
    _fields_ = [('LockObject', _WAITCHAIN_NODE_INFO_STRUCT_1), ('ThreadObject', _WAITCHAIN_NODE_INFO_STRUCT_2)]