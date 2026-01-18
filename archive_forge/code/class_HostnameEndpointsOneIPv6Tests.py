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
class HostnameEndpointsOneIPv6Tests(ClientEndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for the hostname based endpoints when GAI returns only one
    (IPv6) address.
    """

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Creates a L{HostnameEndpoint} instance where the hostname is resolved
        into a single IPv6 address.
        """
        address = HostnameAddress(b'ipv6.example.com', 80)
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(reactor, ['1:2::3:4']), b'ipv6.example.com', address.port, **connectArgs)
        return (endpoint, ('1:2::3:4', address.port, clientFactory, connectArgs.get('timeout', 30), connectArgs.get('bindAddress', None)), address)

    def expectedClients(self, reactor):
        """
        @return: List of calls to L{IReactorTCP.connectTCP}
        """
        return reactor.tcpClients

    def assertConnectArgs(self, receivedArgs, expectedArgs):
        """
        Compare host, port, timeout, and bindAddress in C{receivedArgs}
        to C{expectedArgs}.  We ignore the factory because we don't
        only care what protocol comes out of the
        C{IStreamClientEndpoint.connect} call.

        @param receivedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{timeout}, C{bindAddress}) that was passed to
            L{IReactorTCP.connectTCP}.
        @param expectedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{timeout}, C{bindAddress}) that we expect to have been passed
            to L{IReactorTCP.connectTCP}.
        """
        host, port, ignoredFactory, timeout, bindAddress = receivedArgs
        expectedHost, expectedPort, _ignoredFactory, expectedTimeout, expectedBindAddress = expectedArgs
        self.assertEqual(host, expectedHost)
        self.assertEqual(port, expectedPort)
        self.assertEqual(timeout, expectedTimeout)
        self.assertEqual(bindAddress, expectedBindAddress)

    def connectArgs(self):
        """
        @return: C{dict} of keyword arguments to pass to connect.
        """
        return {'timeout': 10, 'bindAddress': ('localhost', 49595)}

    def test_endpointConnectingCancelled(self):
        """
        Calling L{Deferred.cancel} on the L{Deferred} returned from
        L{IStreamClientEndpoint.connect} is errbacked with an expected
        L{ConnectingCancelledError} exception.
        """
        mreactor = MemoryReactor()
        clientFactory = protocol.Factory()
        clientFactory.protocol = protocol.Protocol
        ep, ignoredArgs, address = self.createClientEndpoint(deterministicResolvingReactor(mreactor, ['127.0.0.1']), clientFactory)
        d = ep.connect(clientFactory)
        d.cancel()
        attemptFactory = self.retrieveConnectedFactory(mreactor)
        attemptFactory.clientConnectionFailed(None, Failure(error.UserError()))
        failure = self.failureResultOf(d)
        self.assertIsInstance(failure.value, error.ConnectingCancelledError)
        self.assertEqual(failure.value.address, address)
        self.assertTrue(mreactor.tcpClients[0][2]._connector.stoppedConnecting)
        self.assertEqual([], mreactor.getDelayedCalls())

    def test_endpointConnectFailure(self):
        """
        If an endpoint tries to connect to a non-listening port it gets
        a C{ConnectError} failure.
        """
        expectedError = error.ConnectError(string='Connection Failed')
        mreactor = RaisingMemoryReactorWithClock(connectException=expectedError)
        clientFactory = object()
        ep, ignoredArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        mreactor.advance(0.3)
        self.assertEqual(self.failureResultOf(d).value, expectedError)
        self.assertEqual([], mreactor.getDelayedCalls())