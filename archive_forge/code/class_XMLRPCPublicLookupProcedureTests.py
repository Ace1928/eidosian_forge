import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
class XMLRPCPublicLookupProcedureTests(unittest.TestCase):
    """
    Tests for L{XMLRPC}'s support of subclasses which override
    C{lookupProcedure} and C{listProcedures}.
    """

    def createServer(self, resource):
        self.p = reactor.listenTCP(0, server.Site(resource), interface='127.0.0.1')
        self.addCleanup(self.p.stopListening)
        self.port = self.p.getHost().port
        self.proxy = xmlrpc.Proxy(networkString('http://127.0.0.1:%d' % self.port))

    def test_lookupProcedure(self):
        """
        A subclass of L{XMLRPC} can override C{lookupProcedure} to find
        procedures that are not defined using a C{xmlrpc_}-prefixed method name.
        """
        self.createServer(TestLookupProcedure())
        what = 'hello'
        d = self.proxy.callRemote('echo', what)
        d.addCallback(self.assertEqual, what)
        return d

    def test_errors(self):
        """
        A subclass of L{XMLRPC} can override C{lookupProcedure} to raise
        L{NoSuchFunction} to indicate that a requested method is not available
        to be called, signalling a fault to the XML-RPC client.
        """
        self.createServer(TestLookupProcedure())
        d = self.proxy.callRemote('xxxx', 'hello')
        d = self.assertFailure(d, xmlrpc.Fault)
        return d

    def test_listMethods(self):
        """
        A subclass of L{XMLRPC} can override C{listProcedures} to define
        Overriding listProcedures should prevent introspection from being
        broken.
        """
        resource = TestListProcedures()
        addIntrospection(resource)
        self.createServer(resource)
        d = self.proxy.callRemote('system.listMethods')

        def listed(procedures):
            self.assertIn('foo', procedures)
        d.addCallback(listed)
        return d