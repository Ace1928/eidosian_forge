from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class RTL_USER_PROCESS_PARAMETERS(Structure):
    _fields_ = [('Reserved1', BYTE * 16), ('Reserved2', PVOID * 10), ('ImagePathName', UNICODE_STRING), ('CommandLine', UNICODE_STRING), ('Environment', PVOID)]