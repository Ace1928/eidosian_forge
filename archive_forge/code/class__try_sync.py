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
class _try_sync:

    def __init__(self, blocking_exc_override: Callable[[BaseException], bool] | None=None):
        self._blocking_exc_override = blocking_exc_override

    def _is_blocking_io_error(self, exc: BaseException) -> bool:
        if self._blocking_exc_override is None:
            return isinstance(exc, BlockingIOError)
        else:
            return self._blocking_exc_override(exc)

    async def __aenter__(self) -> None:
        await trio.lowlevel.checkpoint_if_cancelled()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool:
        if exc_value is not None and self._is_blocking_io_error(exc_value):
            return True
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            return False