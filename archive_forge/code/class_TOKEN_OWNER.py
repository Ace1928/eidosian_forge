from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_OWNER(Structure):
    _fields_ = [('Owner', PSID)]