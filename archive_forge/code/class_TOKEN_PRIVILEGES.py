from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_PRIVILEGES(Structure):
    _fields_ = [('PrivilegeCount', DWORD), ('Privileges', LUID_AND_ATTRIBUTES)]