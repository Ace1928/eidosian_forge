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
class XMLRPCTests(unittest.TestCase):

    def setUp(self):
        self.p = reactor.listenTCP(0, server.Site(Test()), interface='127.0.0.1')
        self.port = self.p.getHost().port
        self.factories = []

    def tearDown(self):
        self.factories = []
        return self.p.stopListening()

    def queryFactory(self, *args, **kwargs):
        """
        Specific queryFactory for proxy that uses our custom
        L{TestQueryFactory}, and save factories.
        """
        factory = TestQueryFactory(*args, **kwargs)
        self.factories.append(factory)
        return factory

    def proxy(self, factory=None):
        """
        Return a new xmlrpc.Proxy for the test site created in
        setUp(), using the given factory as the queryFactory, or
        self.queryFactory if no factory is provided.
        """
        p = xmlrpc.Proxy(networkString('http://127.0.0.1:%d/' % self.port))
        if factory is None:
            p.queryFactory = self.queryFactory
        else:
            p.queryFactory = factory
        return p

    def test_results(self):
        inputOutput = [('add', (2, 3), 5), ('defer', ('a',), 'a'), ('dict', ({'a': 1}, 'a'), 1), ('pair', ('a', 1), ['a', 1]), ('snowman', '☃', '☃'), ('complex', (), {'a': ['b', 'c', 12, []], 'D': 'foo'})]
        dl = []
        for meth, args, outp in inputOutput:
            d = self.proxy().callRemote(meth, *args)
            d.addCallback(self.assertEqual, outp)
            dl.append(d)
        return defer.DeferredList(dl, fireOnOneErrback=True)

    def test_headers(self):
        """
        Verify that headers sent from the client side and the ones we
        get back from the server side are correct.

        """
        d = self.proxy().callRemote('snowman', '☃')

        def check_server_headers(ing):
            self.assertEqual(self.factories[0].headers[b'content-type'], b'text/xml; charset=utf-8')
            self.assertEqual(self.factories[0].headers[b'content-length'], b'129')

        def check_client_headers(ign):
            self.assertEqual(self.factories[0].sent_headers[b'user-agent'], b'Twisted/XMLRPClib')
            self.assertEqual(self.factories[0].sent_headers[b'content-type'], b'text/xml; charset=utf-8')
            self.assertEqual(self.factories[0].sent_headers[b'content-length'], b'155')
        d.addCallback(check_server_headers)
        d.addCallback(check_client_headers)
        return d

    def test_errors(self):
        """
        Verify that for each way a method exposed via XML-RPC can fail, the
        correct 'Content-type' header is set in the response and that the
        client-side Deferred is errbacked with an appropriate C{Fault}
        instance.
        """
        logObserver = EventLoggingObserver()
        filtered = FilteringLogObserver(logObserver, [LogLevelFilterPredicate(defaultLogLevel=LogLevel.critical)])
        globalLogPublisher.addObserver(filtered)
        self.addCleanup(lambda: globalLogPublisher.removeObserver(filtered))
        dl = []
        for code, methodName in [(666, 'fail'), (666, 'deferFail'), (12, 'fault'), (23, 'noSuchMethod'), (17, 'deferFault'), (42, 'SESSION_TEST')]:
            d = self.proxy().callRemote(methodName)
            d = self.assertFailure(d, xmlrpc.Fault)
            d.addCallback(lambda exc, code=code: self.assertEqual(exc.faultCode, code))
            dl.append(d)
        d = defer.DeferredList(dl, fireOnOneErrback=True)

        def cb(ign):
            for factory in self.factories:
                self.assertEqual(factory.headers[b'content-type'], b'text/xml; charset=utf-8')
            self.assertEquals(2, len(logObserver))
            f1 = logObserver[0]['log_failure'].value
            f2 = logObserver[1]['log_failure'].value
            if isinstance(f1, TestValueError):
                self.assertIsInstance(f2, TestRuntimeError)
            else:
                self.assertIsInstance(f1, TestRuntimeError)
                self.assertIsInstance(f2, TestValueError)
            self.flushLoggedErrors(TestRuntimeError, TestValueError)
        d.addCallback(cb)
        return d

    def test_cancel(self):
        """
        A deferred from the Proxy can be cancelled, disconnecting
        the L{twisted.internet.interfaces.IConnector}.
        """

        def factory(*args, **kw):
            factory.f = TestQueryFactoryCancel(*args, **kw)
            return factory.f
        d = self.proxy(factory).callRemote('add', 2, 3)
        self.assertNotEqual(factory.f.connector.state, 'disconnected')
        d.cancel()
        self.assertEqual(factory.f.connector.state, 'disconnected')
        d = self.assertFailure(d, defer.CancelledError)
        return d

    def test_errorGet(self):
        """
        A classic GET on the xml server should return a NOT_ALLOWED.
        """
        agent = client.Agent(reactor)
        d = agent.request(b'GET', networkString('http://127.0.0.1:%d/' % (self.port,)))

        def checkResponse(response):
            self.assertEqual(response.code, http.NOT_ALLOWED)
        d.addCallback(checkResponse)
        return d

    def test_errorXMLContent(self):
        """
        Test that an invalid XML input returns an L{xmlrpc.Fault}.
        """
        agent = client.Agent(reactor)
        d = agent.request(uri=networkString('http://127.0.0.1:%d/' % (self.port,)), method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'foo')))
        d.addCallback(client.readBody)

        def cb(result):
            self.assertRaises(xmlrpc.Fault, xmlrpclib.loads, result)
        d.addCallback(cb)
        return d

    def test_datetimeRoundtrip(self):
        """
        If an L{xmlrpclib.DateTime} is passed as an argument to an XML-RPC
        call and then returned by the server unmodified, the result should
        be equal to the original object.
        """
        when = xmlrpclib.DateTime()
        d = self.proxy().callRemote('echo', when)
        d.addCallback(self.assertEqual, when)
        return d

    def test_doubleEncodingError(self):
        """
        If it is not possible to encode a response to the request (for example,
        because L{xmlrpclib.dumps} raises an exception when encoding a
        L{Fault}) the exception which prevents the response from being
        generated is logged and the request object is finished anyway.
        """
        logObserver = EventLoggingObserver()
        filtered = FilteringLogObserver(logObserver, [LogLevelFilterPredicate(defaultLogLevel=LogLevel.critical)])
        globalLogPublisher.addObserver(filtered)
        self.addCleanup(lambda: globalLogPublisher.removeObserver(filtered))
        d = self.proxy().callRemote('echo', '')

        def fakeDumps(*args, **kwargs):
            raise RuntimeError('Cannot encode anything at all!')
        self.patch(xmlrpclib, 'dumps', fakeDumps)
        d = self.assertFailure(d, Exception)

        def cbFailed(ignored):
            self.assertEquals(1, len(logObserver))
            self.assertIsInstance(logObserver[0]['log_failure'].value, RuntimeError)
            self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
        d.addCallback(cbFailed)
        return d

    def test_closeConnectionAfterRequest(self):
        """
        The connection to the web server is closed when the request is done.
        """
        d = self.proxy().callRemote('echo', '')

        def responseDone(ignored):
            [factory] = self.factories
            self.assertFalse(factory.transport.connected)
            self.assertTrue(factory.transport.disconnected)
        return d.addCallback(responseDone)

    def test_tcpTimeout(self):
        """
        For I{HTTP} URIs, L{xmlrpc.Proxy.callRemote} passes the value it
        received for the C{connectTimeout} parameter as the C{timeout} argument
        to the underlying connectTCP call.
        """
        reactor = MemoryReactor()
        proxy = xmlrpc.Proxy(b'http://127.0.0.1:69', connectTimeout=2.0, reactor=reactor)
        proxy.callRemote('someMethod')
        self.assertEqual(reactor.tcpClients[0][3], 2.0)

    @skipIf(sslSkip, 'OpenSSL not present')
    def test_sslTimeout(self):
        """
        For I{HTTPS} URIs, L{xmlrpc.Proxy.callRemote} passes the value it
        received for the C{connectTimeout} parameter as the C{timeout} argument
        to the underlying connectSSL call.
        """
        reactor = MemoryReactor()
        proxy = xmlrpc.Proxy(b'https://127.0.0.1:69', connectTimeout=3.0, reactor=reactor)
        proxy.callRemote('someMethod')
        self.assertEqual(reactor.sslClients[0][4], 3.0)