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
@skipIf(skipSSL, skipSSLReason)
class SSL4EndpointsTests(EndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for SSL Endpoints.
    """

    def expectedServers(self, reactor):
        """
        @return: List of calls to L{IReactorSSL.listenSSL}
        """
        return reactor.sslServers

    def expectedClients(self, reactor):
        """
        @return: List of calls to L{IReactorSSL.connectSSL}
        """
        return reactor.sslClients

    def assertConnectArgs(self, receivedArgs, expectedArgs):
        """
        Compare host, port, contextFactory, timeout, and bindAddress in
        C{receivedArgs} to C{expectedArgs}.  We ignore the factory because we
        don't only care what protocol comes out of the
        C{IStreamClientEndpoint.connect} call.

        @param receivedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{contextFactory}, C{timeout}, C{bindAddress}) that was passed to
            L{IReactorSSL.connectSSL}.
        @param expectedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{contextFactory}, C{timeout}, C{bindAddress}) that we expect to
            have been passed to L{IReactorSSL.connectSSL}.
        """
        host, port, ignoredFactory, contextFactory, timeout, bindAddress = receivedArgs
        expectedHost, expectedPort, _ignoredFactory, expectedContextFactory, expectedTimeout, expectedBindAddress = expectedArgs
        self.assertEqual(host, expectedHost)
        self.assertEqual(port, expectedPort)
        self.assertEqual(contextFactory, expectedContextFactory)
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

    def setUp(self):
        """
        Set up client and server SSL contexts for use later.
        """
        self.sKey, self.sCert = makeCertificate(O='Server Test Certificate', CN='server')
        self.cKey, self.cCert = makeCertificate(O='Client Test Certificate', CN='client')
        self.serverSSLContext = CertificateOptions(privateKey=self.sKey, certificate=self.sCert, requireCertificate=False)
        self.clientSSLContext = CertificateOptions(requireCertificate=False)

    def createServerEndpoint(self, reactor, factory, **listenArgs):
        """
        Create an L{SSL4ServerEndpoint} and return the tools to verify its
        behaviour.

        @param factory: The thing that we expect to be passed to our
            L{IStreamServerEndpoint.listen} implementation.
        @param reactor: A fake L{IReactorSSL} that L{SSL4ServerEndpoint} can
            call L{IReactorSSL.listenSSL} on.
        @param listenArgs: Optional dictionary of arguments to
            L{IReactorSSL.listenSSL}.
        """
        address = IPv4Address('TCP', '0.0.0.0', 0)
        return (endpoints.SSL4ServerEndpoint(reactor, address.port, self.serverSSLContext, **listenArgs), (address.port, factory, self.serverSSLContext, listenArgs.get('backlog', 50), listenArgs.get('interface', '')), address)

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Create an L{SSL4ClientEndpoint} and return the values needed to verify
        its behaviour.

        @param reactor: A fake L{IReactorSSL} that L{SSL4ClientEndpoint} can
            call L{IReactorSSL.connectSSL} on.
        @param clientFactory: The thing that we expect to be passed to our
            L{IStreamClientEndpoint.connect} implementation.
        @param connectArgs: Optional dictionary of arguments to
            L{IReactorSSL.connectSSL}
        """
        address = IPv4Address('TCP', 'localhost', 80)
        return (endpoints.SSL4ClientEndpoint(reactor, address.host, address.port, self.clientSSLContext, **connectArgs), (address.host, address.port, clientFactory, self.clientSSLContext, connectArgs.get('timeout', 30), connectArgs.get('bindAddress', None)), address)