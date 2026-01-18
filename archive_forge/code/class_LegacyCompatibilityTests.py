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
class LegacyCompatibilityTests(UnitTest):
    """
    Older applications may supply an object to the reactor via
    C{installResolver} that only provides L{IResolverSimple}.
    L{SimpleResolverComplexifier} is a wrapper for an L{IResolverSimple}.
    """

    def test_success(self):
        """
        L{SimpleResolverComplexifier} translates C{resolveHostName} into
        L{IResolutionReceiver.addressResolved}.
        """
        simple = SillyResolverSimple()
        complex = SimpleResolverComplexifier(simple)
        receiver = ResultHolder(self)
        self.assertEqual(receiver._started, False)
        complex.resolveHostName(receiver, 'example.com')
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.assertEqual(receiver._addresses, [])
        simple._requests[0].callback('192.168.1.1')
        self.assertEqual(receiver._addresses, [IPv4Address('TCP', '192.168.1.1', 0)])
        self.assertEqual(receiver._ended, True)

    def test_failure(self):
        """
        L{SimpleResolverComplexifier} translates a known error result from
        L{IResolverSimple.resolveHostName} into an empty result.
        """
        simple = SillyResolverSimple()
        complex = SimpleResolverComplexifier(simple)
        receiver = ResultHolder(self)
        self.assertEqual(receiver._started, False)
        complex.resolveHostName(receiver, 'example.com')
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.assertEqual(receiver._addresses, [])
        simple._requests[0].errback(DNSLookupError('nope'))
        self.assertEqual(receiver._ended, True)
        self.assertEqual(receiver._addresses, [])

    def test_error(self):
        """
        L{SimpleResolverComplexifier} translates an unknown error result from
        L{IResolverSimple.resolveHostName} into an empty result and a logged
        error.
        """
        simple = SillyResolverSimple()
        complex = SimpleResolverComplexifier(simple)
        receiver = ResultHolder(self)
        self.assertEqual(receiver._started, False)
        complex.resolveHostName(receiver, 'example.com')
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.assertEqual(receiver._addresses, [])
        simple._requests[0].errback(ZeroDivisionError('zow'))
        self.assertEqual(len(self.flushLoggedErrors(ZeroDivisionError)), 1)
        self.assertEqual(receiver._ended, True)
        self.assertEqual(receiver._addresses, [])

    def test_simplifier(self):
        """
        L{ComplexResolverSimplifier} translates an L{IHostnameResolver} into an
        L{IResolverSimple} for applications that still expect the old
        interfaces to be in place.
        """
        self.pool, self.doThreadWork = deterministicPool()
        self.reactor, self.doReactorWork = deterministicReactorThreads()
        self.getter = FakeAddrInfoGetter()
        self.resolver = GAIResolver(self.reactor, lambda: self.pool, self.getter.getaddrinfo)
        simpleResolver = ComplexResolverSimplifier(self.resolver)
        self.getter.addResultForHost('example.com', ('192.168.3.4', 4321))
        success = simpleResolver.getHostByName('example.com')
        failure = simpleResolver.getHostByName('nx.example.com')
        self.doThreadWork()
        self.doReactorWork()
        self.doThreadWork()
        self.doReactorWork()
        self.assertEqual(self.failureResultOf(failure).type, DNSLookupError)
        self.assertEqual(self.successResultOf(success), '192.168.3.4')

    def test_portNumber(self):
        """
        L{SimpleResolverComplexifier} preserves the C{port} argument passed to
        C{resolveHostName} in its returned addresses.
        """
        simple = SillyResolverSimple()
        complex = SimpleResolverComplexifier(simple)
        receiver = ResultHolder(self)
        complex.resolveHostName(receiver, 'example.com', 4321)
        self.assertEqual(receiver._started, True)
        self.assertEqual(receiver._ended, False)
        self.assertEqual(receiver._addresses, [])
        simple._requests[0].callback('192.168.1.1')
        self.assertEqual(receiver._addresses, [IPv4Address('TCP', '192.168.1.1', 4321)])
        self.assertEqual(receiver._ended, True)