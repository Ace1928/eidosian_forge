from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
class HostnameResolutionTests(UnitTest):
    """
    Tests for hostname resolution.
    """

    def setUp(self):
        """
        Set up a L{GAIResolver}.
        """
        self.pool, self.doThreadWork = deterministicPool()
        self.reactor, self.doReactorWork = deterministicReactorThreads()
        self.getter = FakeAddrInfoGetter()
        self.resolver = GAIResolver(self.reactor, lambda: self.pool, self.getter.getaddrinfo)

    def test_resolveOneHost(self):
        """
        Resolving an individual hostname that results in one address from
        getaddrinfo results in a single call each to C{resolutionBegan},
        C{addressResolved}, and C{resolutionComplete}.
        """
        receiver = ResultHolder(self)
        self.getter.addResultForHost('sample.example.com', ('4.3.2.1', 0))
        resolution = self.resolver.resolveHostName(receiver, 'sample.example.com')
        self.assertIs(receiver._resolution, resolution)
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.doThreadWork()
        self.doReactorWork()
        self.assertEqual(receiver._ended, True)
        self.assertEqual(receiver._addresses, [IPv4Address('TCP', '4.3.2.1', 0)])

    def test_resolveOneIPv6Host(self):
        """
        Resolving an individual hostname that results in one address from
        getaddrinfo results in a single call each to C{resolutionBegan},
        C{addressResolved}, and C{resolutionComplete}; C{addressResolved} will
        receive an L{IPv6Address}.
        """
        receiver = ResultHolder(self)
        flowInfo = 1
        scopeID = 2
        self.getter.addResultForHost('sample.example.com', ('::1', 0, flowInfo, scopeID), family=AF_INET6)
        resolution = self.resolver.resolveHostName(receiver, 'sample.example.com')
        self.assertIs(receiver._resolution, resolution)
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.doThreadWork()
        self.doReactorWork()
        self.assertEqual(receiver._ended, True)
        self.assertEqual(receiver._addresses, [IPv6Address('TCP', '::1', 0, flowInfo, scopeID)])

    def test_gaierror(self):
        """
        Resolving a hostname that results in C{getaddrinfo} raising a
        L{gaierror} will result in the L{IResolutionReceiver} receiving a call
        to C{resolutionComplete} with no C{addressResolved} calls in between;
        no failure is logged.
        """
        receiver = ResultHolder(self)
        resolution = self.resolver.resolveHostName(receiver, 'sample.example.com')
        self.assertIs(receiver._resolution, resolution)
        self.doThreadWork()
        self.doReactorWork()
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, True)
        self.assertEqual(receiver._addresses, [])

    def _resolveOnlyTest(self, addrTypes, expectedAF):
        """
        Verify that the given set of address types results in the given C{AF_}
        constant being passed to C{getaddrinfo}.

        @param addrTypes: iterable of L{IAddress} implementers

        @param expectedAF: an C{AF_*} constant
        """
        receiver = ResultHolder(self)
        resolution = self.resolver.resolveHostName(receiver, 'sample.example.com', addressTypes=addrTypes)
        self.assertIs(receiver._resolution, resolution)
        self.doThreadWork()
        self.doReactorWork()
        host, port, family, socktype, proto, flags = self.getter.calls[0]
        self.assertEqual(family, expectedAF)

    def test_resolveOnlyIPv4(self):
        """
        When passed an C{addressTypes} parameter containing only
        L{IPv4Address}, L{GAIResolver} will pass C{AF_INET} to C{getaddrinfo}.
        """
        self._resolveOnlyTest([IPv4Address], AF_INET)

    def test_resolveOnlyIPv6(self):
        """
        When passed an C{addressTypes} parameter containing only
        L{IPv6Address}, L{GAIResolver} will pass C{AF_INET6} to C{getaddrinfo}.
        """
        self._resolveOnlyTest([IPv6Address], AF_INET6)

    def test_resolveBoth(self):
        """
        When passed an C{addressTypes} parameter containing both L{IPv4Address}
        and L{IPv6Address} (or the default of C{None}, which carries the same
        meaning), L{GAIResolver} will pass C{AF_UNSPEC} to C{getaddrinfo}.
        """
        self._resolveOnlyTest([IPv4Address, IPv6Address], AF_UNSPEC)
        self._resolveOnlyTest(None, AF_UNSPEC)

    def test_transportSemanticsToSocketType(self):
        """
        When passed a C{transportSemantics} paramter, C{'TCP'} (the value
        present in L{IPv4Address.type} to indicate a stream transport) maps to
        C{SOCK_STREAM} and C{'UDP'} maps to C{SOCK_DGRAM}.
        """
        receiver = ResultHolder(self)
        self.resolver.resolveHostName(receiver, 'example.com', transportSemantics='TCP')
        receiver2 = ResultHolder(self)
        self.resolver.resolveHostName(receiver2, 'example.com', transportSemantics='UDP')
        self.doThreadWork()
        self.doReactorWork()
        self.doThreadWork()
        self.doReactorWork()
        host, port, family, socktypeT, proto, flags = self.getter.calls[0]
        host, port, family, socktypeU, proto, flags = self.getter.calls[1]
        self.assertEqual(socktypeT, SOCK_STREAM)
        self.assertEqual(socktypeU, SOCK_DGRAM)

    def test_socketTypeToAddressType(self):
        """
        When L{GAIResolver} receives a C{SOCK_DGRAM} result from
        C{getaddrinfo}, it returns a C{'TCP'} L{IPv4Address} or L{IPv6Address};
        if it receives C{SOCK_STREAM} then it returns a C{'UDP'} type of same.
        """
        receiver = ResultHolder(self)
        flowInfo = 1
        scopeID = 2
        for socktype in (SOCK_STREAM, SOCK_DGRAM):
            self.getter.addResultForHost('example.com', ('::1', 0, flowInfo, scopeID), family=AF_INET6, socktype=socktype)
            self.getter.addResultForHost('example.com', ('127.0.0.3', 0), family=AF_INET, socktype=socktype)
        self.resolver.resolveHostName(receiver, 'example.com')
        self.doThreadWork()
        self.doReactorWork()
        stream4, stream6, dgram4, dgram6 = receiver._addresses
        self.assertEqual(stream4.type, 'TCP')
        self.assertEqual(stream6.type, 'TCP')
        self.assertEqual(dgram4.type, 'UDP')
        self.assertEqual(dgram6.type, 'UDP')