from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
class HTTP11ClientProtocolTests(TestCase):
    """
    Tests for the HTTP 1.1 client protocol implementation,
    L{HTTP11ClientProtocol}.
    """

    def setUp(self):
        """
        Create an L{HTTP11ClientProtocol} connected to a fake transport.
        """
        self.transport = StringTransport()
        self.protocol = HTTP11ClientProtocol()
        self.protocol.makeConnection(self.transport)

    def test_request(self):
        """
        L{HTTP11ClientProtocol.request} accepts a L{Request} and calls its
        C{writeTo} method with its own transport.
        """
        self.protocol.request(SimpleRequest())
        self.assertEqual(self.transport.value(), b'SOME BYTES')

    def test_secondRequest(self):
        """
        The second time L{HTTP11ClientProtocol.request} is called, it returns a
        L{Deferred} which immediately fires with a L{Failure} wrapping a
        L{RequestNotSent} exception.
        """
        self.protocol.request(SlowRequest())

        def cbNotSent(ignored):
            self.assertEqual(self.transport.value(), b'')
        d = self.assertFailure(self.protocol.request(SimpleRequest()), RequestNotSent)
        d.addCallback(cbNotSent)
        return d

    def test_requestAfterConnectionLost(self):
        """
        L{HTTP11ClientProtocol.request} returns a L{Deferred} which immediately
        fires with a L{Failure} wrapping a L{RequestNotSent} if called after
        the protocol has been disconnected.
        """
        self.protocol.connectionLost(Failure(ConnectionDone('sad transport')))

        def cbNotSent(ignored):
            self.assertEqual(self.transport.value(), b'')
        d = self.assertFailure(self.protocol.request(SimpleRequest()), RequestNotSent)
        d.addCallback(cbNotSent)
        return d

    def test_failedWriteTo(self):
        """
        If the L{Deferred} returned by L{Request.writeTo} fires with a
        L{Failure}, L{HTTP11ClientProtocol.request} disconnects its transport
        and returns a L{Deferred} which fires with a L{Failure} of
        L{RequestGenerationFailed} wrapping the underlying failure.
        """

        class BrokenRequest:
            persistent = False

            def writeTo(self, transport):
                return fail(ArbitraryException())
        d = self.protocol.request(BrokenRequest())

        def cbFailed(ignored):
            self.assertTrue(self.transport.disconnecting)
            self.protocol.connectionLost(Failure(ConnectionDone('you asked for it')))
        d = assertRequestGenerationFailed(self, d, [ArbitraryException])
        d.addCallback(cbFailed)
        return d

    def test_synchronousWriteToError(self):
        """
        If L{Request.writeTo} raises an exception,
        L{HTTP11ClientProtocol.request} returns a L{Deferred} which fires with
        a L{Failure} of L{RequestGenerationFailed} wrapping that exception.
        """

        class BrokenRequest:
            persistent = False

            def writeTo(self, transport):
                raise ArbitraryException()
        d = self.protocol.request(BrokenRequest())
        return assertRequestGenerationFailed(self, d, [ArbitraryException])

    def test_connectionLostDuringRequestGeneration(self, mode=None):
        """
        If L{HTTP11ClientProtocol}'s transport is disconnected before the
        L{Deferred} returned by L{Request.writeTo} fires, the L{Deferred}
        returned by L{HTTP11ClientProtocol.request} fires with a L{Failure} of
        L{RequestTransmissionFailed} wrapping the underlying failure.
        """
        request = SlowRequest()
        d = self.protocol.request(request)
        d = assertRequestTransmissionFailed(self, d, [ArbitraryException])
        self.assertFalse(request.stopped)
        self.protocol.connectionLost(Failure(ArbitraryException()))
        self.assertTrue(request.stopped)
        if mode == 'callback':
            request.finished.callback(None)
        elif mode == 'errback':
            request.finished.errback(Failure(AnotherArbitraryException()))
            errors = self.flushLoggedErrors(AnotherArbitraryException)
            self.assertEqual(len(errors), 1)
        else:
            pass
        return d

    def test_connectionLostBeforeGenerationFinished(self):
        """
        If the request passed to L{HTTP11ClientProtocol} finishes generation
        successfully after the L{HTTP11ClientProtocol}'s connection has been
        lost, nothing happens.
        """
        return self.test_connectionLostDuringRequestGeneration('callback')

    def test_connectionLostBeforeGenerationFailed(self):
        """
        If the request passed to L{HTTP11ClientProtocol} finished generation
        with an error after the L{HTTP11ClientProtocol}'s connection has been
        lost, nothing happens.
        """
        return self.test_connectionLostDuringRequestGeneration('errback')

    def test_errorMessageOnConnectionLostBeforeGenerationFailedDoesNotConfuse(self):
        """
        If the request passed to L{HTTP11ClientProtocol} finished generation
        with an error after the L{HTTP11ClientProtocol}'s connection has been
        lost, an error is logged that gives a non-confusing hint to user on what
        went wrong.
        """
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        def check(ignore):
            self.assertEquals(1, len(logObserver))
            event = logObserver[0]
            self.assertIn('log_failure', event)
            self.assertEqual(event['log_format'], 'Error writing request, but not in valid state to finalize request: {state}')
            self.assertEqual(event['state'], 'CONNECTION_LOST')
        return self.test_connectionLostDuringRequestGeneration('errback').addCallback(check)

    def test_receiveSimplestResponse(self):
        """
        When a response is delivered to L{HTTP11ClientProtocol}, the
        L{Deferred} previously returned by the C{request} method is called back
        with a L{Response} instance and the connection is closed.
        """
        d = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))

        def cbRequest(response):
            self.assertEqual(response.code, 200)
            self.assertEqual(response.headers, Headers())
            self.assertTrue(self.transport.disconnecting)
            self.assertEqual(self.protocol.state, 'QUIESCENT')
        d.addCallback(cbRequest)
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        return d

    def test_receiveResponseHeaders(self):
        """
        The headers included in a response delivered to L{HTTP11ClientProtocol}
        are included on the L{Response} instance passed to the callback
        returned by the C{request} method.
        """
        d = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))

        def cbRequest(response):
            expected = Headers({b'x-foo': [b'bar', b'baz']})
            self.assertEqual(response.headers, expected)
        d.addCallback(cbRequest)
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nX-Foo: baz\r\n\r\n')
        return d

    def test_receiveResponseBeforeRequestGenerationDone(self):
        """
        If response bytes are delivered to L{HTTP11ClientProtocol} before the
        L{Deferred} returned by L{Request.writeTo} fires, those response bytes
        are parsed as part of the response.

        The connection is also closed, because we're in a confusing state, and
        therefore the C{quiescentCallback} isn't called.
        """
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        request = SlowRequest()
        d = protocol.request(request)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nContent-Length: 6\r\n\r\nfoobar')

        def cbResponse(response):
            p = AccumulatingProtocol()
            whenFinished = p.closedDeferred = Deferred()
            response.deliverBody(p)
            self.assertEqual(protocol.state, 'TRANSMITTING_AFTER_RECEIVING_RESPONSE')
            self.assertTrue(transport.disconnecting)
            self.assertEqual(quiescentResult, [])
            return whenFinished.addCallback(lambda ign: (response, p.data))
        d.addCallback(cbResponse)

        def cbAllResponse(result):
            response, body = result
            self.assertEqual(response.version, (b'HTTP', 1, 1))
            self.assertEqual(response.code, 200)
            self.assertEqual(response.phrase, b'OK')
            self.assertEqual(response.headers, Headers({b'x-foo': [b'bar']}))
            self.assertEqual(body, b'foobar')
            request.finished.callback(None)
        d.addCallback(cbAllResponse)
        return d

    def test_receiveResponseHeadersTooLong(self):
        """
        The connection is closed when the server respond with a header which
        is above the maximum line.
        """
        transport = StringTransportWithDisconnection()
        protocol = HTTP11ClientProtocol()
        transport.protocol = protocol
        protocol.makeConnection(transport)
        longLine = b'a' * LineReceiver.MAX_LENGTH
        d = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: ' + longLine + b'\r\nX-Ignored: ignored\r\n\r\n')
        return assertResponseFailed(self, d, [ConnectionDone])

    def test_connectionLostAfterReceivingResponseBeforeRequestGenerationDone(self):
        """
        If response bytes are delivered to L{HTTP11ClientProtocol} before the
        request completes, calling C{connectionLost} on the protocol will
        result in protocol being moved to C{'CONNECTION_LOST'} state.
        """
        request = SlowRequest()
        d = self.protocol.request(request)
        self.protocol.dataReceived(b'HTTP/1.1 400 BAD REQUEST\r\nContent-Length: 9\r\n\r\ntisk tisk')

        def cbResponse(response):
            p = AccumulatingProtocol()
            whenFinished = p.closedDeferred = Deferred()
            response.deliverBody(p)
            return whenFinished.addCallback(lambda ign: (response, p.data))
        d.addCallback(cbResponse)

        def cbAllResponse(ignore):
            request.finished.callback(None)
            self.protocol.connectionLost(Failure(ArbitraryException()))
            self.assertEqual(self.protocol._state, 'CONNECTION_LOST')
        d.addCallback(cbAllResponse)
        return d

    def test_receiveResponseBody(self):
        """
        The C{deliverBody} method of the response object with which the
        L{Deferred} returned by L{HTTP11ClientProtocol.request} fires can be
        used to get the body of the response.
        """
        protocol = AccumulatingProtocol()
        whenFinished = protocol.closedDeferred = Deferred()
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r')
        result = []
        requestDeferred.addCallback(result.append)
        self.assertEqual(result, [])
        self.protocol.dataReceived(b'\n')
        response = result[0]
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'foo')
        self.protocol.dataReceived(b'bar')

        def cbAllResponse(ignored):
            self.assertEqual(protocol.data, b'foobar')
            protocol.closedReason.trap(ResponseDone)
        whenFinished.addCallback(cbAllResponse)
        return whenFinished

    def test_responseBodyFinishedWhenConnectionLostWhenContentLengthIsUnknown(self):
        """
        If the length of the response body is unknown, the protocol passed to
        the response's C{deliverBody} method has its C{connectionLost}
        method called with a L{Failure} wrapping a L{PotentialDataLoss}
        exception.
        """
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'foo')
        self.protocol.dataReceived(b'bar')
        self.assertEqual(protocol.data, b'foobar')
        self.protocol.connectionLost(Failure(ConnectionDone('low-level transport disconnected')))
        protocol.closedReason.trap(PotentialDataLoss)

    def test_chunkedResponseBodyUnfinishedWhenConnectionLost(self):
        """
        If the final chunk has not been received when the connection is lost
        (for any reason), the protocol passed to C{deliverBody} has its
        C{connectionLost} method called with a L{Failure} wrapping the
        exception for that reason.
        """
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'3\r\nfoo\r\n')
        self.protocol.dataReceived(b'3\r\nbar\r\n')
        self.assertEqual(protocol.data, b'foobar')
        self.protocol.connectionLost(Failure(ArbitraryException()))
        return assertResponseFailed(self, fail(protocol.closedReason), [ArbitraryException, _DataLoss])

    def test_parserDataReceivedException(self):
        """
        If the parser L{HTTP11ClientProtocol} delivers bytes to in
        C{dataReceived} raises an exception, the exception is wrapped in a
        L{Failure} and passed to the parser's C{connectionLost} and then the
        L{HTTP11ClientProtocol}'s transport is disconnected.
        """
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'unparseable garbage goes here\r\n')
        d = assertResponseFailed(self, requestDeferred, [ParseError])

        def cbFailed(exc):
            self.assertTrue(self.transport.disconnecting)
            self.assertEqual(exc.reasons[0].value.data, b'unparseable garbage goes here')
            self.protocol.connectionLost(Failure(ConnectionDone('it is done')))
        d.addCallback(cbFailed)
        return d

    def test_proxyStopped(self):
        """
        When the HTTP response parser is disconnected, the
        L{TransportProxyProducer} which was connected to it as a transport is
        stopped.
        """
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        transport = self.protocol._parser.transport
        self.assertIdentical(transport._producer, self.transport)
        self.protocol._disconnectParser(Failure(ConnectionDone('connection done')))
        self.assertIdentical(transport._producer, None)
        return assertResponseFailed(self, requestDeferred, [ConnectionDone])

    def test_abortClosesConnection(self):
        """
        L{HTTP11ClientProtocol.abort} will tell the transport to close its
        connection when it is invoked, and returns a C{Deferred} that fires
        when the connection is lost.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        r1 = []
        r2 = []
        protocol.abort().addCallback(r1.append)
        protocol.abort().addCallback(r2.append)
        self.assertEqual((r1, r2), ([], []))
        self.assertTrue(transport.disconnecting)
        protocol.connectionLost(Failure(ConnectionDone()))
        self.assertEqual(r1, [None])
        self.assertEqual(r2, [None])

    def test_abortAfterConnectionLost(self):
        """
        L{HTTP11ClientProtocol.abort} called after the connection is lost
        returns a C{Deferred} that fires immediately.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        protocol.connectionLost(Failure(ConnectionDone()))
        result = []
        protocol.abort().addCallback(result.append)
        self.assertEqual(result, [None])
        self.assertEqual(protocol._state, 'CONNECTION_LOST')

    def test_abortBeforeResponseBody(self):
        """
        The Deferred returned by L{HTTP11ClientProtocol.request} will fire
        with a L{ResponseFailed} failure containing a L{ConnectionAborted}
        exception, if the connection was aborted before all response headers
        have been received.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.abort()
        self.assertTrue(transport.disconnecting)
        protocol.connectionLost(Failure(ConnectionDone()))
        return assertResponseFailed(self, result, [ConnectionAborted])

    def test_abortAfterResponseHeaders(self):
        """
        When the connection is aborted after the response headers have
        been received and the L{Response} has been made available to
        application code, the response body protocol's C{connectionLost}
        method will be invoked with a L{ResponseFailed} failure containing a
        L{ConnectionAborted} exception.
        """
        transport = StringTransport(lenient=True)
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n')
        testResult = Deferred()

        class BodyDestination(Protocol):
            """
            A body response protocol which immediately aborts the HTTP
            connection.
            """

            def connectionMade(self):
                """
                Abort the HTTP connection.
                """
                protocol.abort()

            def connectionLost(self, reason):
                """
                Make the reason for the losing of the connection available to
                the unit test via C{testResult}.
                """
                testResult.errback(reason)

        def deliverBody(response):
            """
            Connect the L{BodyDestination} response body protocol to the
            response, and then simulate connection loss after ensuring that
            the HTTP connection has been aborted.
            """
            response.deliverBody(BodyDestination())
            self.assertTrue(transport.disconnecting)
            protocol.connectionLost(Failure(ConnectionDone()))

        def checkError(error):
            self.assertIsInstance(error.response, Response)
        result.addCallback(deliverBody)
        deferred = assertResponseFailed(self, testResult, [ConnectionAborted, _DataLoss])
        return deferred.addCallback(checkError)

    def test_quiescentCallbackCalled(self):
        """
        If after a response is done the {HTTP11ClientProtocol} stays open and
        returns to QUIESCENT state, all per-request state is reset and the
        C{quiescentCallback} is called with the protocol instance.

        This is useful for implementing a persistent connection pool.

        The C{quiescentCallback} is called *before* the response-receiving
        protocol's C{connectionLost}, so that new requests triggered by end of
        first request can re-use a persistent connection.
        """
        quiescentResult = []

        def callback(p):
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 3\r\n\r\n')
        self.assertEqual(quiescentResult, [])
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        bodyProtocol.closedDeferred = Deferred()
        bodyProtocol.closedDeferred.addCallback(lambda ign: quiescentResult.append('response done'))
        response.deliverBody(bodyProtocol)
        protocol.dataReceived(b'abc')
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [protocol, 'response done'])
        self.assertEqual(protocol._parser, None)
        self.assertEqual(protocol._finishedRequest, None)
        self.assertEqual(protocol._currentRequest, None)
        self.assertEqual(protocol._transportProxy, None)
        self.assertEqual(protocol._responseDeferred, None)

    def test_transportProducingWhenQuiescentAfterFullBody(self):
        """
        The C{quiescentCallback} passed to L{HTTP11ClientProtocol} will only be
        invoked once that protocol is in a state similar to its initial state.
        One of the aspects of this initial state is the producer-state of its
        transport; an L{HTTP11ClientProtocol} begins with a transport that is
        producing, i.e. not C{pauseProducing}'d.

        Therefore, when C{quiescentCallback} is invoked the protocol will still
        be producing.
        """
        quiescentResult = []

        def callback(p):
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 3\r\n\r\nBBB')
        response = self.successResultOf(requestDeferred)
        self.assertEqual(response._state, 'DEFERRED_CLOSE')
        self.assertEqual(len(quiescentResult), 1)
        self.assertEqual(transport.producerState, 'producing')

    def test_quiescentCallbackCalledEmptyResponse(self):
        """
        The quiescentCallback is called before the request C{Deferred} fires,
        in cases where the response has no body.
        """
        quiescentResult = []

        def callback(p):
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        requestDeferred.addCallback(quiescentResult.append)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        self.assertEqual(len(quiescentResult), 2)
        self.assertIdentical(quiescentResult[0], protocol)
        self.assertIsInstance(quiescentResult[1], Response)

    def test_quiescentCallbackNotCalled(self):
        """
        If after a response is done the {HTTP11ClientProtocol} returns a
        C{Connection: close} header in the response, the C{quiescentCallback}
        is not called and the connection is lost.
        """
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\nConnection: close\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [])
        self.assertTrue(transport.disconnecting)

    def test_quiescentCallbackNotCalledNonPersistentQuery(self):
        """
        If the request was non-persistent (i.e. sent C{Connection: close}),
        the C{quiescentCallback} is not called and the connection is lost.
        """
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=False))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [])
        self.assertTrue(transport.disconnecting)

    def test_quiescentCallbackThrows(self):
        """
        If C{quiescentCallback} throws an exception, the error is logged and
        protocol is disconnected.
        """

        def callback(p):
            raise ZeroDivisionError()
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, ZeroDivisionError)
        self.flushLoggedErrors(ZeroDivisionError)
        self.assertTrue(transport.disconnecting)

    def test_cancelBeforeResponse(self):
        """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{ResponseNeverReceived} failure containing a L{CancelledError}
        exception if the request was cancelled before any response headers were
        received.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        result.cancel()
        self.assertTrue(transport.disconnected)
        return assertWrapperExceptionTypes(self, result, ResponseNeverReceived, [CancelledError])

    def test_cancelDuringResponse(self):
        """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{ResponseFailed} failure containing a L{CancelledError}
        exception if the request was cancelled before all response headers were
        received.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        result.cancel()
        self.assertTrue(transport.disconnected)
        return assertResponseFailed(self, result, [CancelledError])

    def assertCancelDuringBodyProduction(self, producerLength):
        """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{RequestGenerationFailed} failure containing a
        L{CancelledError} exception if the request was cancelled before a
        C{bodyProducer} has finished producing.
        """
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        producer = StringProducer(producerLength)
        nonLocal = {'cancelled': False}

        def cancel(ign):
            nonLocal['cancelled'] = True

        def startProducing(consumer):
            producer.consumer = consumer
            producer.finished = Deferred(cancel)
            return producer.finished
        producer.startProducing = startProducing
        result = protocol.request(Request(b'POST', b'/bar', _boringHeaders, producer))
        producer.consumer.write(b'x' * 5)
        result.cancel()
        self.assertTrue(transport.disconnected)
        self.assertTrue(nonLocal['cancelled'])
        return assertRequestGenerationFailed(self, result, [CancelledError])

    def test_cancelDuringBodyProduction(self):
        """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{RequestGenerationFailed} failure containing a
        L{CancelledError} exception if the request was cancelled before a
        C{bodyProducer} with an explicit length has finished producing.
        """
        return self.assertCancelDuringBodyProduction(10)

    def test_cancelDuringChunkedBodyProduction(self):
        """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{RequestGenerationFailed} failure containing a
        L{CancelledError} exception if the request was cancelled before a
        C{bodyProducer} with C{UNKNOWN_LENGTH} has finished producing.
        """
        return self.assertCancelDuringBodyProduction(UNKNOWN_LENGTH)