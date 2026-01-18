from __future__ import annotations
import errno
import inspect
import os
import socket as stdlib_socket
import sys
import tempfile
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union
import attrs
import pytest
from .. import _core, socket as tsocket
from .._core._tests.tutil import binds_ipv6, creates_ipv6
from .._socket import _NUMERIC_ONLY, SocketType, _SocketType, _try_sync
from ..testing import assert_checkpoints, wait_all_tasks_blocked
class MonkeypatchedGAI:

    def __init__(self, orig_getaddrinfo: Callable[..., GetAddrInfoResponse]) -> None:
        self._orig_getaddrinfo = orig_getaddrinfo
        self._responses: dict[tuple[Any, ...], GetAddrInfoResponse | str] = {}
        self.record: list[tuple[Any, ...]] = []

    def _frozenbind(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        sig = inspect.signature(self._orig_getaddrinfo)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        frozenbound = bound.args
        assert not bound.kwargs
        return frozenbound

    def set(self, response: GetAddrInfoResponse | str, *args: Any, **kwargs: Any) -> None:
        self._responses[self._frozenbind(*args, **kwargs)] = response

    def getaddrinfo(self, *args: Any, **kwargs: Any) -> GetAddrInfoResponse | str:
        bound = self._frozenbind(*args, **kwargs)
        self.record.append(bound)
        if bound in self._responses:
            return self._responses[bound]
        elif bound[-1] & stdlib_socket.AI_NUMERICHOST:
            return self._orig_getaddrinfo(*args, **kwargs)
        else:
            raise RuntimeError(f'gai called with unexpected arguments {bound}')