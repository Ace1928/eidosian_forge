from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorSSL(Interface):

    def connectSSL(host: str, port: int, factory: 'ClientFactory', contextFactory: 'ClientContextFactory', timeout: float, bindAddress: Optional[Tuple[str, int]]) -> IConnector:
        """
        Connect a client Protocol to a remote SSL socket.

        @param host: a host name
        @param port: a port number
        @param factory: a L{twisted.internet.protocol.ClientFactory} instance
        @param contextFactory: a L{twisted.internet.ssl.ClientContextFactory} object.
        @param timeout: number of seconds to wait before assuming the
                        connection has failed.
        @param bindAddress: a (host, port) tuple of local address to bind to,
                            or L{None}.

        @return: An object which provides L{IConnector}.
        """

    def listenSSL(port: int, factory: 'ServerFactory', contextFactory: 'IOpenSSLContextFactory', backlog: int, interface: str) -> 'IListeningPort':
        """
        Connects a given protocol factory to the given numeric TCP/IP port.
        The connection is a SSL one, using contexts created by the context
        factory.

        @param port: a port number on which to listen
        @param factory: a L{twisted.internet.protocol.ServerFactory} instance
        @param contextFactory: an implementor of L{IOpenSSLContextFactory}
        @param backlog: size of the listen queue
        @param interface: the hostname to bind to, defaults to '' (all)
        """