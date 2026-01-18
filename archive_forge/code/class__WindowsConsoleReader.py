import io
import sys
import time
import typing as t
from ctypes import byref
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_ssize_t
from ctypes import c_ulong
from ctypes import c_void_p
from ctypes import POINTER
from ctypes import py_object
from ctypes import Structure
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
from ctypes.wintypes import LPCWSTR
from ctypes.wintypes import LPWSTR
from ._compat import _NonClosingTextIOWrapper
import msvcrt  # noqa: E402
from ctypes import windll  # noqa: E402
from ctypes import WINFUNCTYPE  # noqa: E402
class _WindowsConsoleReader(_WindowsConsoleRawIOBase):

    def readable(self):
        return True

    def readinto(self, b):
        bytes_to_be_read = len(b)
        if not bytes_to_be_read:
            return 0
        elif bytes_to_be_read % 2:
            raise ValueError('cannot read odd number of bytes from UTF-16-LE encoded console')
        buffer = get_buffer(b, writable=True)
        code_units_to_be_read = bytes_to_be_read // 2
        code_units_read = c_ulong()
        rv = ReadConsoleW(HANDLE(self.handle), buffer, code_units_to_be_read, byref(code_units_read), None)
        if GetLastError() == ERROR_OPERATION_ABORTED:
            time.sleep(0.1)
        if not rv:
            raise OSError(f'Windows error: {GetLastError()}')
        if buffer[0] == EOF:
            return 0
        return 2 * code_units_read.value