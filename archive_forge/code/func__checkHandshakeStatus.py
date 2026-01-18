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
def _checkHandshakeStatus(self):
    """
        Ask OpenSSL to proceed with a handshake in progress.

        Initially, this just sends the ClientHello; after some bytes have been
        stuffed in to the C{Connection} object by C{dataReceived}, it will then
        respond to any C{Certificate} or C{KeyExchange} messages.
        """
    if self._aborted:
        return
    try:
        self._tlsConnection.do_handshake()
    except WantReadError:
        self._flushSendBIO()
    except Error:
        self._tlsShutdownFinished(Failure())
    else:
        self._handshakeDone = True
        if IHandshakeListener.providedBy(self.wrappedProtocol):
            self.wrappedProtocol.handshakeCompleted()