import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def _factoriesShouldConnect(clientInfo, serverInfo):
    """
    Should the client and server described by the arguments be connected to
    each other, i.e. do their port numbers match?

    @param clientInfo: the args for connectTCP
    @type clientInfo: L{tuple}

    @param serverInfo: the args for listenTCP
    @type serverInfo: L{tuple}

    @return: If they do match, return factories for the client and server that
        should connect; otherwise return L{None}, indicating they shouldn't be
        connected.
    @rtype: L{None} or 2-L{tuple} of (L{ClientFactory},
        L{IProtocolFactory})
    """
    clientHost, clientPort, clientFactory, clientTimeout, clientBindAddress = clientInfo
    serverPort, serverFactory, serverBacklog, serverInterface = serverInfo
    if serverPort == clientPort:
        return (clientFactory, serverFactory)
    else:
        return None