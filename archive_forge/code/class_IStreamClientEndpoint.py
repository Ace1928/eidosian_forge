from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IStreamClientEndpoint(Interface):
    """
    A stream client endpoint is a place that L{ClientFactory} can connect to.
    For example, a remote TCP host/port pair would be a TCP client endpoint.

    @since: 10.1
    """

    def connect(protocolFactory: IProtocolFactory) -> 'Deferred[IProtocol]':
        """
        Connect the C{protocolFactory} to the location specified by this
        L{IStreamClientEndpoint} provider.

        @param protocolFactory: A provider of L{IProtocolFactory}

        @return: A L{Deferred} that results in an L{IProtocol} upon successful
            connection otherwise a L{Failure} wrapping L{ConnectError} or
            L{NoProtocol <twisted.internet.error.NoProtocol>}.
        """