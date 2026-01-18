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
class TCP4EndpointsTests(EndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for TCP IPv4 Endpoints.
    """

    def expectedServers(self, reactor):
        """
        @return: List of calls to L{IReactorTCP.listenTCP}
        """
        return reactor.tcpServers

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

    def listenArgs(self):
        """
        @return: C{dict} of keyword arguments to pass to listen
        """
        return {'backlog': 100, 'interface': '127.0.0.1'}

    def createServerEndpoint(self, reactor, factory, **listenArgs):
        """
        Create an L{TCP4ServerEndpoint} and return the values needed to verify
        its behaviour.

        @param reactor: A fake L{IReactorTCP} that L{TCP4ServerEndpoint} can
            call L{IReactorTCP.listenTCP} on.
        @param factory: The thing that we expect to be passed to our
            L{IStreamServerEndpoint.listen} implementation.
        @param listenArgs: Optional dictionary of arguments to
            L{IReactorTCP.listenTCP}.
        """
        address = IPv4Address('TCP', '0.0.0.0', 0)
        return (endpoints.TCP4ServerEndpoint(reactor, address.port, **listenArgs), (address.port, factory, listenArgs.get('backlog', 50), listenArgs.get('interface', '')), address)

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Create an L{TCP4ClientEndpoint} and return the values needed to verify
        its behavior.

        @param reactor: A fake L{IReactorTCP} that L{TCP4ClientEndpoint} can
            call L{IReactorTCP.connectTCP} on.
        @param clientFactory: The thing that we expect to be passed to our
            L{IStreamClientEndpoint.connect} implementation.
        @param connectArgs: Optional dictionary of arguments to
            L{IReactorTCP.connectTCP}
        """
        address = IPv4Address('TCP', 'localhost', 80)
        return (endpoints.TCP4ClientEndpoint(reactor, address.host, address.port, **connectArgs), (address.host, address.port, clientFactory, connectArgs.get('timeout', 30), connectArgs.get('bindAddress', None)), address)