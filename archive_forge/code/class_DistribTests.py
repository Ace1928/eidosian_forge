from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
class DistribTests(TestCase):
    port1 = None
    port2 = None
    sub = None
    f1 = None

    def tearDown(self):
        """
        Clean up all the event sources left behind by either directly by
        test methods or indirectly via some distrib API.
        """
        dl = [defer.Deferred(), defer.Deferred()]
        if self.f1 is not None and self.f1.proto is not None:
            self.f1.proto.notifyOnDisconnect(lambda: dl[0].callback(None))
        else:
            dl[0].callback(None)
        if self.sub is not None and self.sub.publisher is not None:
            self.sub.publisher.broker.notifyOnDisconnect(lambda: dl[1].callback(None))
            self.sub.publisher.broker.transport.loseConnection()
        else:
            dl[1].callback(None)
        if self.port1 is not None:
            dl.append(self.port1.stopListening())
        if self.port2 is not None:
            dl.append(self.port2.stopListening())
        return defer.gatherResults(dl)

    def testDistrib(self):
        r1 = resource.Resource()
        r1.putChild(b'there', static.Data(b'root', 'text/plain'))
        site1 = server.Site(r1)
        self.f1 = PBServerFactory(distrib.ResourcePublisher(site1))
        self.port1 = reactor.listenTCP(0, self.f1)
        self.sub = distrib.ResourceSubscription('127.0.0.1', self.port1.getHost().port)
        r2 = resource.Resource()
        r2.putChild(b'here', self.sub)
        f2 = MySite(r2)
        self.port2 = reactor.listenTCP(0, f2)
        agent = client.Agent(reactor)
        url = f'http://127.0.0.1:{self.port2.getHost().port}/here/there'
        url = url.encode('ascii')
        d = agent.request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self.assertEqual, b'root')
        return d

    def _setupDistribServer(self, child):
        """
        Set up a resource on a distrib site using L{ResourcePublisher}.

        @param child: The resource to publish using distrib.

        @return: A tuple consisting of the host and port on which to contact
            the created site.
        """
        distribRoot = resource.Resource()
        distribRoot.putChild(b'child', child)
        distribSite = server.Site(distribRoot)
        self.f1 = distribFactory = PBServerFactory(distrib.ResourcePublisher(distribSite))
        distribPort = reactor.listenTCP(0, distribFactory, interface='127.0.0.1')
        self.addCleanup(distribPort.stopListening)
        addr = distribPort.getHost()
        self.sub = mainRoot = distrib.ResourceSubscription(addr.host, addr.port)
        mainSite = server.Site(mainRoot)
        mainPort = reactor.listenTCP(0, mainSite, interface='127.0.0.1')
        self.addCleanup(mainPort.stopListening)
        mainAddr = mainPort.getHost()
        return (mainPort, mainAddr)

    def _requestTest(self, child, **kwargs):
        """
        Set up a resource on a distrib site using L{ResourcePublisher} and
        then retrieve it from a L{ResourceSubscription} via an HTTP client.

        @param child: The resource to publish using distrib.
        @param **kwargs: Extra keyword arguments to pass to L{Agent.request} when
            requesting the resource.

        @return: A L{Deferred} which fires with the result of the request.
        """
        mainPort, mainAddr = self._setupDistribServer(child)
        agent = client.Agent(reactor)
        url = f'http://{mainAddr.host}:{mainAddr.port}/child'
        url = url.encode('ascii')
        d = agent.request(b'GET', url, **kwargs)
        d.addCallback(client.readBody)
        return d

    def _requestAgentTest(self, child, **kwargs):
        """
        Set up a resource on a distrib site using L{ResourcePublisher} and
        then retrieve it from a L{ResourceSubscription} via an HTTP client.

        @param child: The resource to publish using distrib.
        @param **kwargs: Extra keyword arguments to pass to L{Agent.request} when
            requesting the resource.

        @return: A L{Deferred} which fires with a tuple consisting of a
            L{twisted.test.proto_helpers.AccumulatingProtocol} containing the
            body of the response and an L{IResponse} with the response itself.
        """
        mainPort, mainAddr = self._setupDistribServer(child)
        url = f'http://{mainAddr.host}:{mainAddr.port}/child'
        url = url.encode('ascii')
        d = client.Agent(reactor).request(b'GET', url, **kwargs)

        def cbCollectBody(response):
            protocol = proto_helpers.AccumulatingProtocol()
            response.deliverBody(protocol)
            d = protocol.closedDeferred = defer.Deferred()
            d.addCallback(lambda _: (protocol, response))
            return d
        d.addCallback(cbCollectBody)
        return d

    def test_requestHeaders(self):
        """
        The request headers are available on the request object passed to a
        distributed resource's C{render} method.
        """
        requestHeaders = {}
        logObserver = proto_helpers.EventLoggingObserver()
        globalLogPublisher.addObserver(logObserver)
        req = [None]

        class ReportRequestHeaders(resource.Resource):

            def render(self, request):
                req[0] = request
                requestHeaders.update(dict(request.requestHeaders.getAllRawHeaders()))
                return b''

        def check_logs():
            msgs = [e['log_format'] for e in logObserver]
            self.assertIn('connected to publisher', msgs)
            self.assertIn('could not connect to distributed web service: {msg}', msgs)
            self.assertIn(req[0], msgs)
            globalLogPublisher.removeObserver(logObserver)
        request = self._requestTest(ReportRequestHeaders(), headers=Headers({'foo': ['bar']}))

        def cbRequested(result):
            self.f1.proto.notifyOnDisconnect(check_logs)
            self.assertEqual(requestHeaders[b'Foo'], [b'bar'])
        request.addCallback(cbRequested)
        return request

    def test_requestResponseCode(self):
        """
        The response code can be set by the request object passed to a
        distributed resource's C{render} method.
        """

        class SetResponseCode(resource.Resource):

            def render(self, request):
                request.setResponseCode(200)
                return ''
        request = self._requestAgentTest(SetResponseCode())

        def cbRequested(result):
            self.assertEqual(result[0].data, b'')
            self.assertEqual(result[1].code, 200)
            self.assertEqual(result[1].phrase, b'OK')
        request.addCallback(cbRequested)
        return request

    def test_requestResponseCodeMessage(self):
        """
        The response code and message can be set by the request object passed to
        a distributed resource's C{render} method.
        """

        class SetResponseCode(resource.Resource):

            def render(self, request):
                request.setResponseCode(200, b'some-message')
                return ''
        request = self._requestAgentTest(SetResponseCode())

        def cbRequested(result):
            self.assertEqual(result[0].data, b'')
            self.assertEqual(result[1].code, 200)
            self.assertEqual(result[1].phrase, b'some-message')
        request.addCallback(cbRequested)
        return request

    def test_largeWrite(self):
        """
        If a string longer than the Banana size limit is passed to the
        L{distrib.Request} passed to the remote resource, it is broken into
        smaller strings to be transported over the PB connection.
        """

        class LargeWrite(resource.Resource):

            def render(self, request):
                request.write(b'x' * SIZE_LIMIT + b'y')
                request.finish()
                return server.NOT_DONE_YET
        request = self._requestTest(LargeWrite())
        request.addCallback(self.assertEqual, b'x' * SIZE_LIMIT + b'y')
        return request

    def test_largeReturn(self):
        """
        Like L{test_largeWrite}, but for the case where C{render} returns a
        long string rather than explicitly passing it to L{Request.write}.
        """

        class LargeReturn(resource.Resource):

            def render(self, request):
                return b'x' * SIZE_LIMIT + b'y'
        request = self._requestTest(LargeReturn())
        request.addCallback(self.assertEqual, b'x' * SIZE_LIMIT + b'y')
        return request

    def test_connectionLost(self):
        """
        If there is an error issuing the request to the remote publisher, an
        error response is returned.
        """
        self.f1 = serverFactory = PBServerFactory(pb.Root())
        self.port1 = serverPort = reactor.listenTCP(0, serverFactory)
        self.sub = subscription = distrib.ResourceSubscription('127.0.0.1', serverPort.getHost().port)
        request = DummyRequest([b''])
        d = _render(subscription, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, 500)
            errors = self.flushLoggedErrors(pb.NoSuchMethod)
            self.assertEqual(len(errors), 1)
            expected = [b'', b'<html>', b'  <head><title>500 - Server Connection Lost</title></head>', b'  <body>', b'    <h1>Server Connection Lost</h1>', b'    <p>Connection to distributed server lost:<pre>[Failure instance: Traceback from remote host -- twisted.spread.flavors.NoSuchMethod: No such method: remote_request', b']</pre></p>', b'  </body>', b'</html>', b'']
            self.assertEqual([b'\n'.join(expected)], request.written)
        d.addCallback(cbRendered)
        return d

    def test_logFailed(self):
        """
        When a request fails, the string form of the failure is logged.
        """
        logObserver = proto_helpers.EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        f = failure.Failure(ArbitraryError())
        request = DummyRequest([b''])
        issue = distrib.Issue(request)
        issue.failed(f)
        self.assertEquals(1, len(logObserver))
        self.assertIn('Failure instance', logObserver[0]['log_format'])

    def test_requestFail(self):
        """
        When L{twisted.web.distrib.Request}'s fail is called, the failure
        is logged.
        """
        logObserver = proto_helpers.EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        err = ArbitraryError()
        f = failure.Failure(err)
        req = distrib.Request(DummyChannel())
        req.fail(f)
        self.flushLoggedErrors(ArbitraryError)
        self.assertEquals(1, len(logObserver))
        self.assertIs(logObserver[0]['log_failure'], f)