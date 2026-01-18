from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_APPCONTAINER_INFORMATION(Structure):
    _fields_ = [('TokenAppContainer', PSID)]