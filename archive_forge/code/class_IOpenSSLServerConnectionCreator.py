from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IOpenSSLServerConnectionCreator(Interface):
    """
    A provider of L{IOpenSSLServerConnectionCreator} can create
    L{OpenSSL.SSL.Connection} objects for TLS servers.

    @see: L{twisted.internet.ssl}

    @note: Creating OpenSSL connection objects is subtle, error-prone, and
        security-critical.  Before implementing this interface yourself,
        consider using L{twisted.internet.ssl.CertificateOptions} as your
        C{contextFactory}.  (For historical reasons, that class does not
        actually I{implement} this interface; nevertheless it is usable in all
        Twisted APIs which require a provider of this interface.)
    """

    def serverConnectionForTLS(tlsProtocol: 'TLSMemoryBIOProtocol') -> 'OpenSSLConnection':
        """
        Create a connection for the given server protocol.

        @return: an OpenSSL connection object configured appropriately for the
            given Twisted protocol.
        """