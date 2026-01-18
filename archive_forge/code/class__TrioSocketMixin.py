from __future__ import annotations
import array
import math
import socket
import sys
import types
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from io import IOBase
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind
from types import TracebackType
from typing import (
import trio.from_thread
import trio.lowlevel
from outcome import Error, Outcome, Value
from trio.lowlevel import (
from trio.socket import SocketType as TrioSocketType
from trio.to_thread import run_sync
from .. import CapacityLimiterStatistics, EventStatistics, TaskInfo, abc
from .._core._eventloop import claim_worker_thread
from .._core._exceptions import (
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import CapacityLimiter as BaseCapacityLimiter
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import ResourceGuard
from .._core._tasks import CancelScope as BaseCancelScope
from ..abc import IPSockAddrType, UDPPacketType, UNIXDatagramPacketType
from ..abc._eventloop import AsyncBackend
from ..streams.memory import MemoryObjectSendStream
class _TrioSocketMixin(Generic[T_SockAddr]):

    def __init__(self, trio_socket: TrioSocketType) -> None:
        self._trio_socket = trio_socket
        self._closed = False

    def _check_closed(self) -> None:
        if self._closed:
            raise ClosedResourceError
        if self._trio_socket.fileno() < 0:
            raise BrokenResourceError

    @property
    def _raw_socket(self) -> socket.socket:
        return self._trio_socket._sock

    async def aclose(self) -> None:
        if self._trio_socket.fileno() >= 0:
            self._closed = True
            self._trio_socket.close()

    def _convert_socket_error(self, exc: BaseException) -> NoReturn:
        if isinstance(exc, trio.ClosedResourceError):
            raise ClosedResourceError from exc
        elif self._trio_socket.fileno() < 0 and self._closed:
            raise ClosedResourceError from None
        elif isinstance(exc, OSError):
            raise BrokenResourceError from exc
        else:
            raise exc