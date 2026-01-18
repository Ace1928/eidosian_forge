from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class TestableProxyClientFactory(portforward.ProxyClientFactory):
    """
    Test proxy client factory that keeps the last created protocol instance.

    @ivar protoInstance: the last instance of the protocol.
    @type protoInstance: L{portforward.ProxyClient}
    """

    def buildProtocol(self, addr):
        """
        Create the protocol instance and keeps track of it.
        """
        proto = portforward.ProxyClientFactory.buildProtocol(self, addr)
        self.protoInstance = proto
        return proto