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
@implementer(IOpenSSLClientConnectionCreator, IOpenSSLServerConnectionCreator)
class _ContextFactoryToConnectionFactory:
    """
    Adapter wrapping a L{twisted.internet.interfaces.IOpenSSLContextFactory}
    into a L{IOpenSSLClientConnectionCreator} or
    L{IOpenSSLServerConnectionCreator}.

    See U{https://twistedmatrix.com/trac/ticket/7215} for work that should make
    this unnecessary.
    """

    def __init__(self, oldStyleContextFactory):
        """
        Construct a L{_ContextFactoryToConnectionFactory} with a
        L{twisted.internet.interfaces.IOpenSSLContextFactory}.

        Immediately call C{getContext} on C{oldStyleContextFactory} in order to
        force advance parameter checking, since old-style context factories
        don't actually check that their arguments to L{OpenSSL} are correct.

        @param oldStyleContextFactory: A factory that can produce contexts.
        @type oldStyleContextFactory:
            L{twisted.internet.interfaces.IOpenSSLContextFactory}
        """
        oldStyleContextFactory.getContext()
        self._oldStyleContextFactory = oldStyleContextFactory

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

    def serverConnectionForTLS(self, protocol):
        """
        Construct an OpenSSL server connection from the wrapped old-style
        context factory.

        @note: Since old-style context factories don't distinguish between
            clients and servers, this is exactly the same as
            L{_ContextFactoryToConnectionFactory.clientConnectionForTLS}.

        @param protocol: The protocol initiating a TLS connection.
        @type protocol: L{TLSMemoryBIOProtocol}

        @return: a connection
        @rtype: L{OpenSSL.SSL.Connection}
        """
        return self._connectionForTLS(protocol)

    def clientConnectionForTLS(self, protocol):
        """
        Construct an OpenSSL server connection from the wrapped old-style
        context factory.

        @note: Since old-style context factories don't distinguish between
            clients and servers, this is exactly the same as
            L{_ContextFactoryToConnectionFactory.serverConnectionForTLS}.

        @param protocol: The protocol initiating a TLS connection.
        @type protocol: L{TLSMemoryBIOProtocol}

        @return: a connection
        @rtype: L{OpenSSL.SSL.Connection}
        """
        return self._connectionForTLS(protocol)