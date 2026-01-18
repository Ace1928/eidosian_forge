from winappdbg.win32.defines import *
class _SYSTEM_INFO_OEM_ID(Union):
    _fields_ = [('dwOemId', DWORD), ('w', _SYSTEM_INFO_OEM_ID_STRUCT)]