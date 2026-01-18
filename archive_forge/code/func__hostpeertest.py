from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def _hostpeertest(self, get, testServer):
    """
        Test one of the permutations of client/server host/peer.
        """

    class TestProtocol(Protocol):

        def makeConnection(self, transport):
            Protocol.makeConnection(self, transport)
            self.onConnection.callback(transport)
    if testServer:
        server = TestProtocol()
        d = server.onConnection = Deferred()
        client = Protocol()
    else:
        server = Protocol()
        client = TestProtocol()
        d = client.onConnection = Deferred()
    loopback.loopbackAsync(server, client)

    def connected(transport):
        host = getattr(transport, get)()
        self.assertTrue(IAddress.providedBy(host))
    return d.addCallback(connected)