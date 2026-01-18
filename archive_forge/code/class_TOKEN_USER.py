from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_USER(Structure):
    _fields_ = [('User', SID_AND_ATTRIBUTES)]