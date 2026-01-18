from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
class ProcessEndpointTransportTests(unittest.TestCase):
    """
    Test the behaviour of the implementation detail
    L{endpoints._ProcessEndpointTransport}.
    """

    def setUp(self):
        self.reactor = MemoryProcessReactor()
        self.endpoint = endpoints.ProcessEndpoint(self.reactor, b'/bin/executable')
        protocol = self.successResultOf(self.endpoint.connect(Factory.forProtocol(Protocol)))
        self.process = self.reactor.processTransport
        self.endpointTransport = protocol.transport

    def test_verifyConsumer(self):
        """
        L{_ProcessEndpointTransport}s provide L{IConsumer}.
        """
        verifyObject(IConsumer, self.endpointTransport)

    def test_verifyProducer(self):
        """
        L{_ProcessEndpointTransport}s provide L{IPushProducer}.
        """
        verifyObject(IPushProducer, self.endpointTransport)

    def test_verifyTransport(self):
        """
        L{_ProcessEndpointTransport}s provide L{ITransport}.
        """
        verifyObject(ITransport, self.endpointTransport)

    def test_constructor(self):
        """
        The L{_ProcessEndpointTransport} instance stores the process passed to
        it.
        """
        self.assertIs(self.endpointTransport._process, self.process)

    def test_registerProducer(self):
        """
        Registering a producer with the endpoint transport registers it with
        the underlying process transport.
        """

        @implementer(IPushProducer)
        class AProducer:
            pass
        aProducer = AProducer()
        self.endpointTransport.registerProducer(aProducer, False)
        self.assertIs(self.process.producer, aProducer)

    def test_pauseProducing(self):
        """
        Pausing the endpoint transport pauses the underlying process transport.
        """
        self.endpointTransport.pauseProducing()
        self.assertEqual(self.process.producerState, 'paused')

    def test_resumeProducing(self):
        """
        Resuming the endpoint transport resumes the underlying process
        transport.
        """
        self.test_pauseProducing()
        self.endpointTransport.resumeProducing()
        self.assertEqual(self.process.producerState, 'producing')

    def test_stopProducing(self):
        """
        Stopping the endpoint transport as a producer stops the underlying
        process transport.
        """
        self.endpointTransport.stopProducing()
        self.assertEqual(self.process.producerState, 'stopped')

    def test_unregisterProducer(self):
        """
        Unregistring the endpoint transport's producer unregisters the
        underlying process transport's producer.
        """
        self.test_registerProducer()
        self.endpointTransport.unregisterProducer()
        self.assertIsNone(self.process.producer)

    def test_extraneousAttributes(self):
        """
        L{endpoints._ProcessEndpointTransport} filters out extraneous
        attributes of its underlying transport, to present a more consistent
        cross-platform view of subprocesses and prevent accidental
        dependencies.
        """
        self.process.pipes = []
        self.assertRaises(AttributeError, getattr, self.endpointTransport, 'pipes')

    def test_writeSequence(self):
        """
        The writeSequence method of L{_ProcessEndpointTransport} writes a list
        of string passed to it to the transport's stdin.
        """
        self.endpointTransport.writeSequence([b'test1', b'test2', b'test3'])
        self.assertEqual(self.process.io.getvalue(), b'test1test2test3')

    def test_write(self):
        """
        The write method of L{_ProcessEndpointTransport} writes a string of
        data passed to it to the child process's stdin.
        """
        self.endpointTransport.write(b'test')
        self.assertEqual(self.process.io.getvalue(), b'test')

    def test_loseConnection(self):
        """
        A call to the loseConnection method of a L{_ProcessEndpointTransport}
        instance returns a call to the process transport's loseConnection.
        """
        self.endpointTransport.loseConnection()
        self.assertFalse(self.process.connected)

    def test_getHost(self):
        """
        L{_ProcessEndpointTransport.getHost} returns a L{_ProcessAddress}
        instance matching the process C{getHost}.
        """
        host = self.endpointTransport.getHost()
        self.assertIsInstance(host, _ProcessAddress)
        self.assertIs(host, self.process.getHost())

    def test_getPeer(self):
        """
        L{_ProcessEndpointTransport.getPeer} returns a L{_ProcessAddress}
        instance matching the process C{getPeer}.
        """
        peer = self.endpointTransport.getPeer()
        self.assertIsInstance(peer, _ProcessAddress)
        self.assertIs(peer, self.process.getPeer())