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
def _make_simple_sock_method_wrapper(fn: Callable[Concatenate[_stdlib_socket.socket, P], T], wait_fn: Callable[[_stdlib_socket.socket], Awaitable[None]], maybe_avail: bool=False) -> Callable[Concatenate[_SocketType, P], Awaitable[T]]:

    @_wraps(fn, assigned=('__name__',), updated=())
    async def wrapper(self: _SocketType, *args: P.args, **kwargs: P.kwargs) -> T:
        return await self._nonblocking_helper(wait_fn, fn, *args, **kwargs)
    wrapper.__doc__ = f'Like :meth:`socket.socket.{fn.__name__}`, but async.\n\n            '
    if maybe_avail:
        wrapper.__doc__ += f'Only available on platforms where :meth:`socket.socket.{fn.__name__}` is available.'
    return wrapper