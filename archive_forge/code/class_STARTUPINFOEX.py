import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class STARTUPINFOEX(Structure):
    _fields_ = [('StartupInfo', STARTUPINFO), ('lpAttributeList', PPROC_THREAD_ATTRIBUTE_LIST)]