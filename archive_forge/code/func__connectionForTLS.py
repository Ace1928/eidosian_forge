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
def _connectionForTLS(self, protocol):
    """
        Create an L{OpenSSL.SSL.Connection} object.

        @param protocol: The protocol initiating a TLS connection.
        @type protocol: L{TLSMemoryBIOProtocol}

        @return: a connection
        @rtype: L{OpenSSL.SSL.Connection}
        """
    context = self._oldStyleContextFactory.getContext()
    return Connection(context, None)