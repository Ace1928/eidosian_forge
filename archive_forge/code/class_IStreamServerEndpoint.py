from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IStreamServerEndpoint(Interface):
    """
    A stream server endpoint is a place that a L{Factory} can listen for
    incoming connections.

    @since: 10.1
    """

    def listen(protocolFactory: IProtocolFactory) -> 'Deferred[IListeningPort]':
        """
        Listen with C{protocolFactory} at the location specified by this
        L{IStreamServerEndpoint} provider.

        @param protocolFactory: A provider of L{IProtocolFactory}

        @return: A L{Deferred} that results in an L{IListeningPort} or an
            L{CannotListenError}
        """