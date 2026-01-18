from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
class QueryTests(SynchronousTestCase):
    """
    Tests for L{ResolverBase.query}.
    """

    def test_resolverBaseProvidesIResolver(self):
        """
        L{ResolverBase} provides the L{IResolver} interface.
        """
        verifyClass(IResolver, ResolverBase)

    def test_typeToMethodDispatch(self):
        """
        L{ResolverBase.query} looks up a method to invoke using the type of the
        query passed to it and the C{typeToMethod} mapping on itself.
        """
        results = []
        resolver = ResolverBase()
        resolver.typeToMethod = {12345: lambda query, timeout: results.append((query, timeout))}
        query = Query(name=b'example.com', type=12345)
        resolver.query(query, 123)
        self.assertEqual([(b'example.com', 123)], results)

    def test_typeToMethodResult(self):
        """
        L{ResolverBase.query} returns a L{Deferred} which fires with the result
        of the method found in the C{typeToMethod} mapping for the type of the
        query passed to it.
        """
        expected = object()
        resolver = ResolverBase()
        resolver.typeToMethod = {54321: lambda query, timeout: expected}
        query = Query(name=b'example.com', type=54321)
        queryDeferred = resolver.query(query, 123)
        result = []
        queryDeferred.addBoth(result.append)
        self.assertEqual(expected, result[0])

    def test_unknownQueryType(self):
        """
        L{ResolverBase.query} returns a L{Deferred} which fails with
        L{NotImplementedError} when called with a query of a type not present in
        its C{typeToMethod} dictionary.
        """
        resolver = ResolverBase()
        resolver.typeToMethod = {}
        query = Query(name=b'example.com', type=12345)
        queryDeferred = resolver.query(query, 123)
        result = []
        queryDeferred.addBoth(result.append)
        self.assertIsInstance(result[0], Failure)
        result[0].trap(NotImplementedError)