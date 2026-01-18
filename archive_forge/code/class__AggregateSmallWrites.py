from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
class _AggregateSmallWrites:
    """
    Aggregate small writes so they get written in large batches.

    If this is used as part of a transport, the transport needs to call
    ``flush()`` immediately when ``loseConnection()`` is called, otherwise any
    buffered writes will never get written.

    @cvar MAX_BUFFER_SIZE: The maximum amount of bytes to buffer before writing
        them out.
    """
    MAX_BUFFER_SIZE = 64000

    def __init__(self, write: Callable[[bytes], object], clock: IReactorTime):
        self._write = write
        self._clock = clock
        self._buffer: list[bytes] = []
        self._bufferLeft = self.MAX_BUFFER_SIZE
        self._scheduled: Optional[IDelayedCall] = None

    def write(self, data: bytes) -> None:
        """
        Buffer the data, or write it immediately if we've accumulated enough to
        make it worth it.

        Accumulating too much data can result in higher memory usage.
        """
        self._buffer.append(data)
        self._bufferLeft -= len(data)
        if self._bufferLeft < 0:
            self.flush()
            return
        if self._scheduled:
            return
        self._scheduled = self._clock.callLater(0, self._scheduledFlush)

    def _scheduledFlush(self) -> None:
        """Called in next reactor iteration."""
        self._scheduled = None
        self.flush()

    def flush(self) -> None:
        """Flush any buffered writes."""
        if self._buffer:
            self._bufferLeft = self.MAX_BUFFER_SIZE
            self._write(b''.join(self._buffer))
            del self._buffer[:]