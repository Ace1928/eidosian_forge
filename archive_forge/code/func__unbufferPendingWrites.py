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
def _unbufferPendingWrites(self):
    """
        Un-buffer all waiting writes in L{TLSMemoryBIOProtocol._appSendBuffer}.
        """
    pendingWrites, self._appSendBuffer = (self._appSendBuffer, [])
    for eachWrite in pendingWrites:
        self._write(eachWrite)
    if self._appSendBuffer:
        return
    if self._producer is not None:
        self._producer.resumeProducing()
        return
    if self.disconnecting:
        self._shutdownTLS()