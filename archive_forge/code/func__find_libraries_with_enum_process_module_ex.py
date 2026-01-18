import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def _find_libraries_with_enum_process_module_ex(self):
    """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on windows system only.
        This code is adapted from code by Philipp Hagemeister @phihag available
        at https://stackoverflow.com/questions/17474574
        """
    from ctypes.wintypes import DWORD, HMODULE, MAX_PATH
    PROCESS_QUERY_INFORMATION = 1024
    PROCESS_VM_READ = 16
    LIST_LIBRARIES_ALL = 3
    ps_api = self._get_windll('Psapi')
    kernel_32 = self._get_windll('kernel32')
    h_process = kernel_32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, os.getpid())
    if not h_process:
        raise OSError(f'Could not open PID {os.getpid()}')
    try:
        buf_count = 256
        needed = DWORD()
        while True:
            buf = (HMODULE * buf_count)()
            buf_size = ctypes.sizeof(buf)
            if not ps_api.EnumProcessModulesEx(h_process, ctypes.byref(buf), buf_size, ctypes.byref(needed), LIST_LIBRARIES_ALL):
                raise OSError('EnumProcessModulesEx failed')
            if buf_size >= needed.value:
                break
            buf_count = needed.value // (buf_size // buf_count)
        count = needed.value // (buf_size // buf_count)
        h_modules = map(HMODULE, buf[:count])
        buf = ctypes.create_unicode_buffer(MAX_PATH)
        n_size = DWORD()
        for h_module in h_modules:
            if not ps_api.GetModuleFileNameExW(h_process, h_module, ctypes.byref(buf), ctypes.byref(n_size)):
                raise OSError('GetModuleFileNameEx failed')
            filepath = buf.value
            self._make_controller_from_path(filepath)
    finally:
        kernel_32.CloseHandle(h_process)