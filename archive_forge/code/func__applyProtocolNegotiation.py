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
def _applyProtocolNegotiation(self, connection):
    """
        Applies ALPN/NPN protocol neogitation to the connection, if the factory
        supports it.

        @param connection: The OpenSSL connection object to have ALPN/NPN added
            to it.
        @type connection: L{OpenSSL.SSL.Connection}

        @return: Nothing
        @rtype: L{None}
        """
    if IProtocolNegotiationFactory.providedBy(self.wrappedFactory):
        protocols = self.wrappedFactory.acceptableProtocols()
        context = connection.get_context()
        _setAcceptableProtocols(context, protocols)
    return