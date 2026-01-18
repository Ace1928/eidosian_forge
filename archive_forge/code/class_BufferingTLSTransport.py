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
class BufferingTLSTransport(TLSMemoryBIOProtocol):
    """
    A TLS transport implemented by wrapping buffering around a
    L{TLSMemoryBIOProtocol}.

    Doing many small writes directly to a L{OpenSSL.SSL.Connection}, as
    implemented in L{TLSMemoryBIOProtocol}, can add significant CPU and
    bandwidth overhead.  Thus, even when writing is possible, small writes will
    get aggregated and written as a single write at the next reactor iteration.
    """

    def __init__(self, factory: TLSMemoryBIOFactory, wrappedProtocol: IProtocol, _connectWrapped: bool=True):
        super().__init__(factory, wrappedProtocol, _connectWrapped)
        actual_write = super().write
        self._aggregator = _AggregateSmallWrites(actual_write, factory._clock)
        self.write = self._aggregator.write

    def writeSequence(self, sequence: Iterable[bytes]) -> None:
        self._aggregator.write(b''.join(sequence))

    def loseConnection(self) -> None:
        self._aggregator.flush()
        super().loseConnection()