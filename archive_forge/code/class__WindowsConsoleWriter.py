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
class _WindowsConsoleWriter(_WindowsConsoleRawIOBase):

    def writable(self):
        return True

    @staticmethod
    def _get_error_message(errno):
        if errno == ERROR_SUCCESS:
            return 'ERROR_SUCCESS'
        elif errno == ERROR_NOT_ENOUGH_MEMORY:
            return 'ERROR_NOT_ENOUGH_MEMORY'
        return f'Windows error {errno}'

    def write(self, b):
        bytes_to_be_written = len(b)
        buf = get_buffer(b)
        code_units_to_be_written = min(bytes_to_be_written, MAX_BYTES_WRITTEN) // 2
        code_units_written = c_ulong()
        WriteConsoleW(HANDLE(self.handle), buf, code_units_to_be_written, byref(code_units_written), None)
        bytes_written = 2 * code_units_written.value
        if bytes_written == 0 and bytes_to_be_written > 0:
            raise OSError(self._get_error_message(GetLastError()))
        return bytes_written