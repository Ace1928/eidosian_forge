import contextlib
import ctypes
import os
from ctypes.wintypes import (
from shellingham._core import SHELL_NAMES
class ProcessEntry32(ctypes.Structure):
    _fields_ = (('dwSize', DWORD), ('cntUsage', DWORD), ('th32ProcessID', DWORD), ('th32DefaultHeapID', ctypes.POINTER(ULONG)), ('th32ModuleID', DWORD), ('cntThreads', DWORD), ('th32ParentProcessID', DWORD), ('pcPriClassBase', LONG), ('dwFlags', DWORD), ('szExeFile', CHAR * MAX_PATH))