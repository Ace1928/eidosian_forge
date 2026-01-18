import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class VS_FIXEDFILEINFO(Structure):
    _fields_ = [('dwSignature', DWORD), ('dwStrucVersion', DWORD), ('dwFileVersionMS', DWORD), ('dwFileVersionLS', DWORD), ('dwProductVersionMS', DWORD), ('dwProductVersionLS', DWORD), ('dwFileFlagsMask', DWORD), ('dwFileFlags', DWORD), ('dwFileOS', DWORD), ('dwFileType', DWORD), ('dwFileSubtype', DWORD), ('dwFileDateMS', DWORD), ('dwFileDateLS', DWORD)]