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
class AdoptedStreamServerEndpointTests(ServerEndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for adopted socket-based stream server endpoints.
    """

    def _createStubbedAdoptedEndpoint(self, reactor, fileno, addressFamily):
        """
        Create an L{AdoptedStreamServerEndpoint} which may safely be used with
        an invalid file descriptor.  This is convenient for a number of unit
        tests.
        """
        e = endpoints.AdoptedStreamServerEndpoint(reactor, fileno, addressFamily)
        e._close = lambda fd: None
        e._setNonBlocking = lambda fd: None
        return e

    def createServerEndpoint(self, reactor, factory):
        """
        Create a new L{AdoptedStreamServerEndpoint} for use by a test.

        @return: A three-tuple:
            - The endpoint
            - A tuple of the arguments expected to be passed to the underlying
              reactor method
            - An IAddress object which will match the result of
              L{IListeningPort.getHost} on the port returned by the endpoint.
        """
        fileno = 12
        addressFamily = AF_INET
        endpoint = self._createStubbedAdoptedEndpoint(reactor, fileno, addressFamily)
        address = IPv4Address('TCP', '0.0.0.0', 1234)
        return (endpoint, (fileno, addressFamily, factory), address)

    def expectedServers(self, reactor):
        """
        @return: The ports which were actually adopted by C{reactor} via calls
            to its L{IReactorSocket.adoptStreamPort} implementation.
        """
        return reactor.adoptedPorts

    def listenArgs(self):
        """
        @return: A C{dict} of additional keyword arguments to pass to the
            C{createServerEndpoint}.
        """
        return {}

    def test_singleUse(self):
        """
        L{AdoptedStreamServerEndpoint.listen} can only be used once.  The file
        descriptor given is closed after the first use, and subsequent calls to
        C{listen} return a L{Deferred} that fails with L{AlreadyListened}.
        """
        reactor = MemoryReactor()
        endpoint = self._createStubbedAdoptedEndpoint(reactor, 13, AF_INET)
        endpoint.listen(object())
        d = self.assertFailure(endpoint.listen(object()), error.AlreadyListened)

        def listenFailed(ignored):
            self.assertEqual(1, len(reactor.adoptedPorts))
        d.addCallback(listenFailed)
        return d

    def test_descriptionNonBlocking(self):
        """
        L{AdoptedStreamServerEndpoint.listen} sets the file description given
        to it to non-blocking.
        """
        reactor = MemoryReactor()
        endpoint = self._createStubbedAdoptedEndpoint(reactor, 13, AF_INET)
        events = []

        def setNonBlocking(fileno):
            events.append(('setNonBlocking', fileno))
        endpoint._setNonBlocking = setNonBlocking
        d = endpoint.listen(object())

        def listened(ignored):
            self.assertEqual([('setNonBlocking', 13)], events)
        d.addCallback(listened)
        return d

    def test_descriptorClosed(self):
        """
        L{AdoptedStreamServerEndpoint.listen} closes its file descriptor after
        adding it to the reactor with L{IReactorSocket.adoptStreamPort}.
        """
        reactor = MemoryReactor()
        endpoint = self._createStubbedAdoptedEndpoint(reactor, 13, AF_INET)
        events = []

        def close(fileno):
            events.append(('close', fileno, len(reactor.adoptedPorts)))
        endpoint._close = close
        d = endpoint.listen(object())

        def listened(ignored):
            self.assertEqual([('close', 13, 1)], events)
        d.addCallback(listened)
        return d