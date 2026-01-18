from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorTCP(Interface):

    def listenTCP(port: int, factory: 'ServerFactory', backlog: int, interface: str) -> 'IListeningPort':
        """
        Connects a given protocol factory to the given numeric TCP/IP port.

        @param port: a port number on which to listen
        @param factory: a L{twisted.internet.protocol.ServerFactory} instance
        @param backlog: size of the listen queue
        @param interface: The local IPv4 or IPv6 address to which to bind;
            defaults to '', ie all IPv4 addresses.  To bind to all IPv4 and IPv6
            addresses, you must call this method twice.

        @return: an object that provides L{IListeningPort}.

        @raise CannotListenError: as defined here
                                  L{twisted.internet.error.CannotListenError},
                                  if it cannot listen on this port (e.g., it
                                  cannot bind to the required port number)
        """

    def connectTCP(host: str, port: int, factory: 'ClientFactory', timeout: float, bindAddress: Optional[Tuple[str, int]]) -> IConnector:
        """
        Connect a TCP client.

        @param host: A hostname or an IPv4 or IPv6 address literal.
        @param port: a port number
        @param factory: a L{twisted.internet.protocol.ClientFactory} instance
        @param timeout: number of seconds to wait before assuming the
                        connection has failed.
        @param bindAddress: a (host, port) tuple of local address to bind
                            to, or None.

        @return: An object which provides L{IConnector}. This connector will
                 call various callbacks on the factory when a connection is
                 made, failed, or lost - see
                 L{ClientFactory<twisted.internet.protocol.ClientFactory>}
                 docs for details.
        """