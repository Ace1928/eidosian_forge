from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
class BootstrapTests(SynchronousTestCase):
    """
    Tests for L{root.bootstrap}
    """

    def test_returnsDeferredResolver(self):
        """
        L{root.bootstrap} returns an object which is initially a
        L{root.DeferredResolver}.
        """
        deferredResolver = root.bootstrap(StubResolver())
        self.assertIsInstance(deferredResolver, root.DeferredResolver)

    def test_resolves13RootServers(self):
        """
        The L{IResolverSimple} supplied to L{root.bootstrap} is used to lookup
        the IP addresses of the 13 root name servers.
        """
        stubResolver = StubResolver()
        root.bootstrap(stubResolver)
        self.assertEqual(stubResolver.calls, [((s,), {}) for s in ROOT_SERVERS])

    def test_becomesResolver(self):
        """
        The L{root.DeferredResolver} initially returned by L{root.bootstrap}
        becomes a L{root.Resolver} when the supplied resolver has successfully
        looked up all root hints.
        """
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertIsInstance(deferredResolver, Resolver)

    def test_resolverReceivesRootHints(self):
        """
        The L{root.Resolver} which eventually replaces L{root.DeferredResolver}
        is supplied with the IP addresses of the 13 root servers.
        """
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 13)

    def test_continuesWhenSomeRootHintsFail(self):
        """
        The L{root.Resolver} is eventually created, even if some of the root
        hint lookups fail. Only the working root hint IP addresses are supplied
        to the L{root.Resolver}.
        """
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        results = iter(stubResolver.pendingResults)
        d1 = next(results)
        for d in results:
            d.callback('192.0.2.101')
        d1.errback(TimeoutError())

        def checkHints(res):
            self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 12)
        d1.addBoth(checkHints)

    def test_continuesWhenAllRootHintsFail(self):
        """
        The L{root.Resolver} is eventually created, even if all of the root hint
        lookups fail. Pending and new lookups will then fail with
        AttributeError.
        """
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        results = iter(stubResolver.pendingResults)
        d1 = next(results)
        for d in results:
            d.errback(TimeoutError())
        d1.errback(TimeoutError())

        def checkHints(res):
            self.assertEqual(deferredResolver.hints, [])
        d1.addBoth(checkHints)
        self.addCleanup(self.flushLoggedErrors, TimeoutError)

    def test_passesResolverFactory(self):
        """
        L{root.bootstrap} accepts a C{resolverFactory} argument which is passed
        as an argument to L{root.Resolver} when it has successfully looked up
        root hints.
        """
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver, resolverFactory=raisingResolverFactory)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertIs(deferredResolver._resolverFactory, raisingResolverFactory)