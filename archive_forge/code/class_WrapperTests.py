import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class WrapperTests(unittest.TestCase):
    """
    Tests for L{WrappingFactory} and L{ProtocolWrapper}.
    """

    def test_protocolFactoryAttribute(self):
        """
        Make sure protocol.factory is the wrapped factory, not the wrapping
        factory.
        """
        f = Server()
        wf = policies.WrappingFactory(f)
        p = wf.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
        self.assertIs(p.wrappedProtocol.factory, f)

    def test_transportInterfaces(self):
        """
        The transport wrapper passed to the wrapped protocol's
        C{makeConnection} provides the same interfaces as are provided by the
        original transport.
        """

        class IStubTransport(Interface):
            pass

        @implementer(IStubTransport)
        class StubTransport:
            pass
        implementedBy(policies.ProtocolWrapper)
        proto = protocol.Protocol()
        wrapper = policies.ProtocolWrapper(policies.WrappingFactory(None), proto)
        wrapper.makeConnection(StubTransport())
        self.assertTrue(IStubTransport.providedBy(proto.transport))

    def test_factoryLogPrefix(self):
        """
        L{WrappingFactory.logPrefix} is customized to mention both the original
        factory and the wrapping factory.
        """
        server = Server()
        factory = policies.WrappingFactory(server)
        self.assertEqual('Server (WrappingFactory)', factory.logPrefix())

    def test_factoryLogPrefixFallback(self):
        """
        If the wrapped factory doesn't have a L{logPrefix} method,
        L{WrappingFactory.logPrefix} falls back to the factory class name.
        """

        class NoFactory:
            pass
        server = NoFactory()
        factory = policies.WrappingFactory(server)
        self.assertEqual('NoFactory (WrappingFactory)', factory.logPrefix())

    def test_protocolLogPrefix(self):
        """
        L{ProtocolWrapper.logPrefix} is customized to mention both the original
        protocol and the wrapper.
        """
        server = Server()
        factory = policies.WrappingFactory(server)
        protocol = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
        self.assertEqual('EchoProtocol (ProtocolWrapper)', protocol.logPrefix())

    def test_protocolLogPrefixFallback(self):
        """
        If the wrapped protocol doesn't have a L{logPrefix} method,
        L{ProtocolWrapper.logPrefix} falls back to the protocol class name.
        """

        class NoProtocol:
            pass
        server = Server()
        server.protocol = NoProtocol
        factory = policies.WrappingFactory(server)
        protocol = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
        self.assertEqual('NoProtocol (ProtocolWrapper)', protocol.logPrefix())

    def _getWrapper(self):
        """
        Return L{policies.ProtocolWrapper} that has been connected to a
        L{StringTransport}.
        """
        wrapper = policies.ProtocolWrapper(policies.WrappingFactory(Server()), protocol.Protocol())
        transport = StringTransport()
        wrapper.makeConnection(transport)
        return wrapper

    def test_getHost(self):
        """
        L{policies.ProtocolWrapper.getHost} calls C{getHost} on the underlying
        transport.
        """
        wrapper = self._getWrapper()
        self.assertEqual(wrapper.getHost(), wrapper.transport.getHost())

    def test_getPeer(self):
        """
        L{policies.ProtocolWrapper.getPeer} calls C{getPeer} on the underlying
        transport.
        """
        wrapper = self._getWrapper()
        self.assertEqual(wrapper.getPeer(), wrapper.transport.getPeer())

    def test_registerProducer(self):
        """
        L{policies.ProtocolWrapper.registerProducer} calls C{registerProducer}
        on the underlying transport.
        """
        wrapper = self._getWrapper()
        producer = object()
        wrapper.registerProducer(producer, True)
        self.assertIs(wrapper.transport.producer, producer)
        self.assertTrue(wrapper.transport.streaming)

    def test_unregisterProducer(self):
        """
        L{policies.ProtocolWrapper.unregisterProducer} calls
        C{unregisterProducer} on the underlying transport.
        """
        wrapper = self._getWrapper()
        producer = object()
        wrapper.registerProducer(producer, True)
        wrapper.unregisterProducer()
        self.assertIsNone(wrapper.transport.producer)
        self.assertIsNone(wrapper.transport.streaming)

    def test_stopConsuming(self):
        """
        L{policies.ProtocolWrapper.stopConsuming} calls C{stopConsuming} on
        the underlying transport.
        """
        wrapper = self._getWrapper()
        result = []
        wrapper.transport.stopConsuming = lambda: result.append(True)
        wrapper.stopConsuming()
        self.assertEqual(result, [True])

    def test_startedConnecting(self):
        """
        L{policies.WrappingFactory.startedConnecting} calls
        C{startedConnecting} on the underlying factory.
        """
        result = []

        class Factory:

            def startedConnecting(self, connector):
                result.append(connector)
        wrapper = policies.WrappingFactory(Factory())
        connector = object()
        wrapper.startedConnecting(connector)
        self.assertEqual(result, [connector])

    def test_clientConnectionLost(self):
        """
        L{policies.WrappingFactory.clientConnectionLost} calls
        C{clientConnectionLost} on the underlying factory.
        """
        result = []

        class Factory:

            def clientConnectionLost(self, connector, reason):
                result.append((connector, reason))
        wrapper = policies.WrappingFactory(Factory())
        connector = object()
        reason = object()
        wrapper.clientConnectionLost(connector, reason)
        self.assertEqual(result, [(connector, reason)])

    def test_clientConnectionFailed(self):
        """
        L{policies.WrappingFactory.clientConnectionFailed} calls
        C{clientConnectionFailed} on the underlying factory.
        """
        result = []

        class Factory:

            def clientConnectionFailed(self, connector, reason):
                result.append((connector, reason))
        wrapper = policies.WrappingFactory(Factory())
        connector = object()
        reason = object()
        wrapper.clientConnectionFailed(connector, reason)
        self.assertEqual(result, [(connector, reason)])

    def test_breakReferenceCycle(self):
        """
        L{policies.ProtocolWrapper.connectionLost} sets C{wrappedProtocol} to
        C{None} in order to break reference cycle between wrapper and wrapped
        protocols.
        :return:
        """
        wrapper = policies.ProtocolWrapper(policies.WrappingFactory(Server()), protocol.Protocol())
        transport = StringTransportWithDisconnection()
        transport.protocol = wrapper
        wrapper.makeConnection(transport)
        self.assertIsNotNone(wrapper.wrappedProtocol)
        transport.loseConnection()
        self.assertIsNone(wrapper.wrappedProtocol)