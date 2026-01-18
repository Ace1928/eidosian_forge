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
class _SignalReceiver:
    _iterator: AsyncIterator[int]

    def __init__(self, signals: tuple[Signals, ...]):
        self._signals = signals

    def __enter__(self) -> _SignalReceiver:
        self._cm = trio.open_signal_receiver(*self._signals)
        self._iterator = self._cm.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def __aiter__(self) -> _SignalReceiver:
        return self

    async def __anext__(self) -> Signals:
        signum = await self._iterator.__anext__()
        return Signals(signum)