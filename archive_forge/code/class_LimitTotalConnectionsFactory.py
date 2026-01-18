import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class LimitTotalConnectionsFactory(ServerFactory):
    """
    Factory that limits the number of simultaneous connections.

    @type connectionCount: C{int}
    @ivar connectionCount: number of current connections.
    @type connectionLimit: C{int} or L{None}
    @cvar connectionLimit: maximum number of connections.
    @type overflowProtocol: L{Protocol} or L{None}
    @cvar overflowProtocol: Protocol to use for new connections when
        connectionLimit is exceeded.  If L{None} (the default value), excess
        connections will be closed immediately.
    """
    connectionCount = 0
    connectionLimit = None
    overflowProtocol: Optional[Type[Protocol]] = None

    def buildProtocol(self, addr):
        if self.connectionLimit is None or self.connectionCount < self.connectionLimit:
            wrappedProtocol = self.protocol()
        elif self.overflowProtocol is None:
            return None
        else:
            wrappedProtocol = self.overflowProtocol()
        wrappedProtocol.factory = self
        protocol = ProtocolWrapper(self, wrappedProtocol)
        self.connectionCount += 1
        return protocol

    def registerProtocol(self, p):
        pass

    def unregisterProtocol(self, p):
        self.connectionCount -= 1