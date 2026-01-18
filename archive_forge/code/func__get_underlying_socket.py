from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
def _get_underlying_socket(sock: _HasFileNo | int | Handle, *, which: WSAIoctls=WSAIoctls.SIO_BASE_HANDLE) -> Handle:
    if hasattr(sock, 'fileno'):
        sock = sock.fileno()
    base_ptr = ffi.new('HANDLE *')
    out_size = ffi.new('DWORD *')
    failed = ws2_32.WSAIoctl(ffi.cast('SOCKET', sock), which, ffi.NULL, 0, base_ptr, ffi.sizeof('HANDLE'), out_size, ffi.NULL, ffi.NULL)
    if failed:
        code = ws2_32.WSAGetLastError()
        raise_winerror(code)
    return Handle(base_ptr[0])