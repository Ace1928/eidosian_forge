import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
class ConnectionCompleter:
    """
    A L{ConnectionCompleter} can cause synthetic TCP connections established by
    L{MemoryReactor.connectTCP} and L{MemoryReactor.listenTCP} to succeed or
    fail.
    """

    def __init__(self, memoryReactor):
        """
        Create a L{ConnectionCompleter} from a L{MemoryReactor}.

        @param memoryReactor: The reactor to attach to.
        @type memoryReactor: L{MemoryReactor}
        """
        self._reactor = memoryReactor

    def succeedOnce(self, debug=False):
        """
        Complete a single TCP connection established on this
        L{ConnectionCompleter}'s L{MemoryReactor}.

        @param debug: A flag; whether to dump output from the established
            connection to stdout.
        @type debug: L{bool}

        @return: a pump for the connection, or L{None} if no connection could
            be established.
        @rtype: L{IOPump} or L{None}
        """
        memoryReactor = self._reactor
        for clientIdx, clientInfo in enumerate(memoryReactor.tcpClients):
            for serverInfo in memoryReactor.tcpServers:
                factories = _factoriesShouldConnect(clientInfo, serverInfo)
                if factories:
                    memoryReactor.tcpClients.remove(clientInfo)
                    memoryReactor.connectors.pop(clientIdx)
                    clientFactory, serverFactory = factories
                    clientProtocol = clientFactory.buildProtocol(None)
                    serverProtocol = serverFactory.buildProtocol(None)
                    serverTransport = makeFakeServer(serverProtocol)
                    clientTransport = makeFakeClient(clientProtocol)
                    return connect(serverProtocol, serverTransport, clientProtocol, clientTransport, debug)

    def failOnce(self, reason=Failure(ConnectionRefusedError())):
        """
        Fail a single TCP connection established on this
        L{ConnectionCompleter}'s L{MemoryReactor}.

        @param reason: the reason to provide that the connection failed.
        @type reason: L{Failure}
        """
        self._reactor.tcpClients.pop(0)[2].clientConnectionFailed(self._reactor.connectors.pop(0), reason)