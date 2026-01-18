from __future__ import annotations
import os
import select
import socket as _stdlib_socket
import sys
from operator import index
from socket import AddressFamily, SocketKind
from typing import (
import idna as _idna
import trio
from trio._util import wraps as _wraps
from . import _core
def _is_blocking_io_error(self, exc: BaseException) -> bool:
    if self._blocking_exc_override is None:
        return isinstance(exc, BlockingIOError)
    else:
        return self._blocking_exc_override(exc)