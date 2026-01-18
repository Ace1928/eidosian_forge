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
class HostnameEndpointMemoryIPv4ReactorTests(_HostnameEndpointMemoryReactorMixin, unittest.TestCase):
    """
    IPv4 resolution tests for L{HostnameEndpoint} with
    L{MemoryReactor} subclasses that do not provide
    L{IReactorPluggableNameResolver}.
    """

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Creates a L{HostnameEndpoint} instance where the hostname is
        resolved into a single IPv4 address.

        @param reactor: The L{MemoryReactor}

        @param clientFactory: The client L{IProtocolFactory}

        @param connectArgs: Additional arguments to
            L{HostnameEndpoint.connect}

        @return: A L{tuple} of the form C{(endpoint, (expectedAddress,
            expectedPort, clientFactory, timeout, localBindAddress,
            hostnameAddress))}
        """
        expectedAddress = '1.2.3.4'
        address = HostnameAddress(b'example.com', 80)
        endpoint = endpoints.HostnameEndpoint(reactor, b'example.com', address.port, **connectArgs)

        def fakegetaddrinfo(host, port, family, socktype):
            return [(AF_INET, SOCK_STREAM, IPPROTO_TCP, '', (expectedAddress, 80))]
        endpoint._getaddrinfo = fakegetaddrinfo
        endpoint._deferToThread = self.synchronousDeferredToThread
        return (endpoint, (expectedAddress, address.port, clientFactory, connectArgs.get('timeout', 30), connectArgs.get('bindAddress', None)), address)