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
class SimpleHostnameResolverTests(unittest.SynchronousTestCase):
    """
    Tests for L{endpoints._SimpleHostnameResolver}.

    @ivar fakeResolverCalls: Arguments with which L{fakeResolver} was
        called.
    @type fakeResolverCalls: L{list} of C{(hostName, port)} L{tuple}s.

    @ivar fakeResolverReturns: The return value of L{fakeResolver}.
    @type fakeResolverReturns: L{Deferred}

    @ivar resolver: The instance to test.
    @type resolver: L{endpoints._SimpleHostnameResolver}

    @ivar resolutionBeganCalls: Arguments with which receiver's
        C{resolutionBegan} method was called.
    @type resolutionBeganCalls: L{list}

    @ivar addressResolved: Arguments with which C{addressResolved} was
        called.
    @type addressResolved: L{list}

    @ivar resolutionCompleteCallCount: The number of calls to the
        receiver's C{resolutionComplete} method.
    @type resolutionCompleteCallCount: L{int}

    @ivar receiver: A L{interfaces.IResolutionReceiver} provider.
    """

    def setUp(self):
        self.fakeResolverCalls = []
        self.fakeResolverReturns = defer.Deferred()
        self.resolver = endpoints._SimpleHostnameResolver(self.fakeResolver)
        self.resolutionBeganCalls = []
        self.addressResolvedCalls = []
        self.resolutionCompleteCallCount = 0

        @provider(interfaces.IResolutionReceiver)
        class _Receiver:

            @staticmethod
            def resolutionBegan(resolutionInProgress):
                self.resolutionBeganCalls.append(resolutionInProgress)

            @staticmethod
            def addressResolved(address):
                self.addressResolvedCalls.append(address)

            @staticmethod
            def resolutionComplete():
                self.resolutionCompleteCallCount += 1
        self.receiver = _Receiver

    def fakeResolver(self, hostName, portNumber):
        """
        A fake resolver callable.

        @param hostName: The hostname to resolve.

        @param portNumber: The port number the returned address should
            include.

        @return: L{fakeResolverCalls}
        @rtype: L{Deferred}
        """
        self.fakeResolverCalls.append((hostName, portNumber))
        return self.fakeResolverReturns

    def test_interface(self):
        """
        A L{endpoints._SimpleHostnameResolver} instance provides
        L{interfaces.IHostnameResolver}.
        """
        self.assertTrue(verifyObject(interfaces.IHostnameResolver, self.resolver))

    def test_resolveNameFailure(self):
        """
        A resolution failure is logged with the name that failed to
        resolve and the callable that tried to resolve it.  The
        resolution receiver begins, receives no addresses, and
        completes.
        """
        logs = []

        @provider(ILogObserver)
        def captureLogs(event):
            logs.append(event)
        globalLogPublisher.addObserver(captureLogs)
        self.addCleanup(lambda: globalLogPublisher.removeObserver(captureLogs))
        self.resolver.resolveHostName(self.receiver, 'example.com')
        self.fakeResolverReturns.errback(Exception())
        self.assertEqual(1, len(logs))
        self.assertEqual(1, len(self.flushLoggedErrors(Exception)))
        [event] = logs
        self.assertTrue(event.get('isError'))
        self.assertTrue(event.get('name', 'example.com'))
        self.assertTrue(event.get('callable', repr(self.fakeResolver)))
        self.assertEqual(1, len(self.resolutionBeganCalls))
        self.assertEqual(self.resolutionBeganCalls[0].name, 'example.com')
        self.assertFalse(self.addressResolvedCalls)
        self.assertEqual(1, self.resolutionCompleteCallCount)

    def test_resolveNameDelivers(self):
        """
        The resolution receiver begins, and resolved hostnames are
        delivered before it completes.
        """
        port = 80
        ipv4Host = '1.2.3.4'
        ipv6Host = '1::2::3::4'
        self.resolver.resolveHostName(self.receiver, 'example.com')
        self.fakeResolverReturns.callback([(AF_INET, SOCK_STREAM, IPPROTO_TCP, '', (ipv4Host, port)), (AF_INET6, SOCK_STREAM, IPPROTO_TCP, '', (ipv6Host, port))])
        self.assertEqual(1, len(self.resolutionBeganCalls))
        self.assertEqual(self.resolutionBeganCalls[0].name, 'example.com')
        self.assertEqual(self.addressResolvedCalls, [IPv4Address('TCP', ipv4Host, port), IPv6Address('TCP', ipv6Host, port)])
        self.assertEqual(self.resolutionCompleteCallCount, 1)