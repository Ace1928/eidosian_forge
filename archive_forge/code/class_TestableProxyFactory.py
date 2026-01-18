from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class TestableProxyFactory(portforward.ProxyFactory):
    """
    Test proxy factory that keeps the last created protocol instance.

    @ivar protoInstance: the last instance of the protocol.
    @type protoInstance: L{portforward.ProxyServer}

    @ivar clientFactoryInstance: client factory used by C{protoInstance} to
        create forward connections.
    @type clientFactoryInstance: L{TestableProxyClientFactory}
    """

    def buildProtocol(self, addr):
        """
        Create the protocol instance, keeps track of it, and makes it use
        C{clientFactoryInstance} as client factory.
        """
        proto = portforward.ProxyFactory.buildProtocol(self, addr)
        self.clientFactoryInstance = TestableProxyClientFactory()
        proto.clientProtocolFactory = lambda: self.clientFactoryInstance
        self.protoInstance = proto
        return proto