import contextlib
import ctypes
import os
from ctypes.wintypes import (
from shellingham._core import SHELL_NAMES
def _iter_processes():
    f = kernel32.CreateToolhelp32Snapshot
    with _handle(f, TH32CS_SNAPPROCESS, 0) as snap:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        ret = kernel32.Process32First(snap, entry)
        while ret:
            yield entry
            ret = kernel32.Process32Next(snap, entry)