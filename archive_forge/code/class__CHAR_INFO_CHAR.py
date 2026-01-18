import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class _CHAR_INFO_CHAR(Union):
    _fields_ = [('UnicodeChar', WCHAR), ('AsciiChar', CHAR)]