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
def _get_text_stdout(buffer_stream: t.BinaryIO) -> t.TextIO:
    text_stream = _NonClosingTextIOWrapper(io.BufferedWriter(_WindowsConsoleWriter(STDOUT_HANDLE)), 'utf-16-le', 'strict', line_buffering=True)
    return t.cast(t.TextIO, ConsoleStream(text_stream, buffer_stream))