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
def _get_windows_console_stream(f: t.TextIO, encoding: t.Optional[str], errors: t.Optional[str]) -> t.Optional[t.TextIO]:
    if get_buffer is not None and encoding in {'utf-16-le', None} and (errors in {'strict', None}) and _is_console(f):
        func = _stream_factories.get(f.fileno())
        if func is not None:
            b = getattr(f, 'buffer', None)
            if b is None:
                return None
            return func(b)