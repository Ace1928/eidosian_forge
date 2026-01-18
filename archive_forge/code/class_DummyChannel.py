from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class DummyChannel:
    """
    A dummy HTTP channel, that does nothing but holds a transport and saves
    connection lost.

    @ivar transport: the transport used by the client.
    @ivar lostReason: the reason saved at connection lost.
    """

    def __init__(self, transport):
        """
        Hold a reference to the transport.
        """
        self.transport = transport
        self.lostReason = None

    def connectionLost(self, reason):
        """
        Keep track of the connection lost reason.
        """
        self.lostReason = reason

    def getPeer(self):
        """
        Get peer information from the transport.
        """
        return self.transport.getPeer()

    def getHost(self):
        """
        Get host information from the transport.
        """
        return self.transport.getHost()