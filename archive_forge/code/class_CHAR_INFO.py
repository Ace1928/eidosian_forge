import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class CHAR_INFO(Structure):
    _fields_ = [('Char', _CHAR_INFO_CHAR), ('Attributes', WORD)]