import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
class HTTP2ServerTests(unittest.TestCase, HTTP2TestHelpers):
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]
    postRequestHeaders = [(b':method', b'POST'), (b':authority', b'localhost'), (b':path', b'/post_endpoint'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'content-length', b'25')]
    postRequestData = [b'hello ', b'world, ', b"it's ", b'http/2!']
    getResponseHeaders = [(b':status', b'200'), (b'request', b'/'), (b'command', b'GET'), (b'version', b'HTTP/2'), (b'content-length', b'13')]
    getResponseData = b"'''\nNone\n'''\n"
    postResponseHeaders = [(b':status', b'200'), (b'request', b'/post_endpoint'), (b'command', b'POST'), (b'version', b'HTTP/2'), (b'content-length', b'36')]
    postResponseData = b"'''\n25\nhello world, it's http/2!'''\n"

    def connectAndReceive(self, connection, headers, body):
        """
        Takes a single L{H2Connection} object and connects it to a
        L{StringTransport} using a brand new L{FrameFactory}.

        @param connection: The L{H2Connection} object to connect.
        @type connection: L{H2Connection}

        @param headers: The headers to send on the first request.
        @type headers: L{Iterable} of L{tuple} of C{(bytes, bytes)}

        @param body: Chunks of body to send, if any.
        @type body: L{Iterable} of L{bytes}

        @return: A tuple of L{FrameFactory}, L{StringTransport}
        """
        frameFactory = FrameFactory()
        transport = StringTransport()
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += buildRequestBytes(headers, body, frameFactory)
        connection.makeConnection(transport)
        for byte in iterbytes(requestBytes):
            connection.dataReceived(byte)
        return (frameFactory, transport)

    def test_basicRequest(self):
        """
        Send request over a TCP connection and confirm that we get back the
        expected data in the order and style we expect.
        """
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[3], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[1].data), dict(self.getResponseHeaders))
            self.assertEqual(frames[2].data, self.getResponseData)
            self.assertEqual(frames[3].data, b'')
            self.assertTrue('END_STREAM' in frames[3].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_postRequest(self):
        """
        Send a POST request and confirm that the data is safely transferred.
        """
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        _, transport = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
            self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[-3].data), dict(self.postResponseHeaders))
            self.assertEqual(frames[-2].data, self.postResponseData)
            self.assertEqual(frames[-1].data, b'')
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_postRequestNoLength(self):
        """
        Send a POST request without length and confirm that the data is safely
        transferred.
        """
        postResponseHeaders = [(b':status', b'200'), (b'request', b'/post_endpoint'), (b'command', b'POST'), (b'version', b'HTTP/2'), (b'content-length', b'38')]
        postResponseData = b"'''\nNone\nhello world, it's http/2!'''\n"
        postRequestHeaders = [(x, y) for x, y in self.postRequestHeaders if x != b'content-length']
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        _, transport = self.connectAndReceive(connection, postRequestHeaders, self.postRequestData)

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
            self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[-3].data), dict(postResponseHeaders))
            self.assertEqual(frames[-2].data, postResponseData)
            self.assertEqual(frames[-1].data, b'')
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_interleavedRequests(self):
        """
        Many interleaved POST requests all get received and responded to
        appropriately.
        """
        REQUEST_COUNT = 40
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        streamIDs = list(range(1, REQUEST_COUNT * 2, 2))
        frames = [buildRequestFrames(self.postRequestHeaders, self.postRequestData, f, streamID) for streamID in streamIDs]
        requestBytes = f.clientConnectionPreface()
        frames = itertools.chain.from_iterable(zip(*frames))
        requestBytes += b''.join((frame.serialize() for frame in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(results):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 1 + 3 * 40)
            for streamID in streamIDs:
                streamFrames = [f for f in frames if f.stream_id == streamID and (not isinstance(f, hyperframe.frame.WindowUpdateFrame))]
                self.assertEqual(len(streamFrames), 3)
                self.assertEqual(dict(streamFrames[0].data), dict(self.postResponseHeaders))
                self.assertEqual(streamFrames[1].data, self.postResponseData)
                self.assertEqual(streamFrames[2].data, b'')
                self.assertTrue('END_STREAM' in streamFrames[2].flags)
        return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)

    def test_sendAccordingToPriority(self):
        """
        Data in responses is interleaved according to HTTP/2 priorities.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = ChunkedHTTPHandlerProxy
        getRequestHeaders = self.getRequestHeaders
        getRequestHeaders[2] = (':path', '/chunked/4')
        frames = [buildRequestFrames(getRequestHeaders, [], f, streamID) for streamID in [1, 3, 5]]
        frames[0][0].flags.add('PRIORITY')
        frames[0][0].stream_weight = 64
        frames[1][0].flags.add('PRIORITY')
        frames[1][0].stream_weight = 32
        priorityFrame = f.buildPriorityFrame(streamID=5, weight=16, dependsOn=1, exclusive=True)
        frames[2].insert(0, priorityFrame)
        frames = itertools.chain.from_iterable(frames)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((frame.serialize() for frame in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(results):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 19)
            streamIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            expectedOrder = [1, 3, 1, 1, 3, 1, 1, 3, 5, 3, 5, 3, 5, 5, 5]
            self.assertEqual(streamIDs, expectedOrder)
        return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)

    def test_protocolErrorTerminatesConnection(self):
        """
        A protocol error from the remote peer terminates the connection.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        requestBytes += f.buildPushPromiseFrame(streamID=1, promisedStreamID=2, headers=self.getRequestHeaders, flags=['END_HEADERS']).serialize()
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
            if b.disconnecting:
                break
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 3)
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
        self.assertTrue(b.disconnecting)

    def test_streamProducingData(self):
        """
        The H2Stream data implements IPushProducer, and can have its data
        production controlled by the Request if the Request chooses to.
        """
        connection = H2Connection()
        connection.requestFactory = ConsumerDummyHandlerProxy
        _, transport = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)
        request = connection.streams[1]._request.original
        self.assertFalse(request._requestReceived)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        request.acceptData()
        self.assertTrue(request._requestReceived)
        self.assertTrue(request._data, b"hello world, it's http/2!")
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_abortStreamProducingData(self):
        """
        The H2Stream data implements IPushProducer, and can have its data
        production controlled by the Request if the Request chooses to.
        When the production is stopped, that causes the stream connection to
        be lost.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = AbortingConsumerDummyHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        frames[-1].flags = set()
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        request = a.streams[1]._request.original
        self.assertFalse(request._requestReceived)
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.acceptData()

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 2)
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.RstStreamFrame))
            self.assertEqual(frames[-1].stream_id, 1)
        return cleanupCallback.addCallback(validate)

    def test_terminatedRequest(self):
        """
        When a RstStream frame is received, the L{H2Connection} and L{H2Stream}
        objects tear down the L{http.Request} and swallow all outstanding
        writes.
        """
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        request.write(b'first chunk')
        request.write(b'second chunk')
        cleanupCallback = connection._streamCleanupCallbacks[1]
        connection.dataReceived(frameFactory.buildRstStreamFrame(1, errorCode=1).serialize())
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)
        request.write(b'third chunk')

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[1].stream_id, 1)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        return cleanupCallback.addCallback(validate)

    def test_terminatedConnection(self):
        """
        When a GoAway frame is received, the L{H2Connection} and L{H2Stream}
        objects tear down all outstanding L{http.Request} objects and stop all
        writing.
        """
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        request.write(b'first chunk')
        request.write(b'second chunk')
        cleanupCallback = connection._streamCleanupCallbacks[1]
        connection.dataReceived(frameFactory.buildGoAwayFrame(lastStreamID=0).serialize())
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)
        self.assertFalse(connection._stillProducing)

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[1].stream_id, 1)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        return cleanupCallback.addCallback(validate)

    def test_respondWith100Continue(self):
        """
        Requests containing Expect: 100-continue cause provisional 100
        responses to be emitted.
        """
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        headers = self.getRequestHeaders + [(b'expect', b'100-continue')]
        _, transport = self.connectAndReceive(connection, headers, [])

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 5)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertEqual(frames[1].data, [(b':status', b'100')])
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_respondWith400(self):
        """
        Triggering the call to L{H2Stream._respondToBadRequestAndDisconnect}
        leads to a 400 error being sent automatically and the stream being torn
        down.
        """
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        cleanupCallback = connection._streamCleanupCallbacks[1]
        stream._respondToBadRequestAndDisconnect()
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertEqual(frames[1].data, [(b':status', b'400')])
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return cleanupCallback.addCallback(validate)

    def test_loseH2StreamConnection(self):
        """
        Calling L{Request.loseConnection} causes all data that has previously
        been sent to be flushed, and then the stream cleanly closed.
        """
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        dataChunks = [b'hello', b'world', b'here', b'are', b'some', b'writes']
        for chunk in dataChunks:
            request.write(chunk)
        request.loseConnection()

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 9)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertTrue('END_STREAM' in frames[-1].flags)
            receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(receivedDataChunks, dataChunks + [b''])
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_cannotRegisterTwoProducers(self):
        """
        The L{H2Stream} object forbids registering two producers.
        """
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        self.assertRaises(ValueError, stream.registerProducer, request, True)

    def test_handlesPullProducer(self):
        """
        L{Request} objects that have registered pull producers get blocked and
        unblocked according to HTTP/2 flow control.
        """
        connection = H2Connection()
        connection.requestFactory = DummyPullProducerHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        producerComplete = request._actualProducer.result
        producerComplete.addCallback(lambda x: request.finish())

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b''])
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_isSecureWorksProperly(self):
        """
        L{Request} objects can correctly ask isSecure on HTTP/2.
        """
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        self.assertFalse(request.isSecure())
        connection.streams[1].abortConnection()

    def test_lateCompletionWorks(self):
        """
        L{H2Connection} correctly unblocks when a stream is ended.
        """
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        reactor.callLater(0.01, request.finish)

        def validateComplete(*args):
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validateComplete)

    def test_writeSequenceForChannels(self):
        """
        L{H2Stream} objects can send a series of frames via C{writeSequence}.
        """
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        request.setResponseCode(200)
        stream.writeSequence([b'Hello', b',', b'world!'])
        request.finish()
        completionDeferred = connection._streamCleanupCallbacks[1]

        def validate(streamID):
            frames = framesFromBytes(transport.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'Hello', b',', b'world!', b''])
        return completionDeferred.addCallback(validate)

    def test_delayWrites(self):
        """
        Delaying writes from L{Request} causes the L{H2Connection} to block on
        sending until data is available. However, data is *not* sent if there's
        no room in the flow control window.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DelayedHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        request.write(b'fiver')
        dataChunks = [b'here', b'are', b'some', b'writes']

        def write_chunks():
            for chunk in dataChunks:
                request.write(chunk)
            request.finish()
        d = task.deferLater(reactor, 0.01, write_chunks)
        d.addCallback(lambda *args: a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize()))

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 9)
            self.assertTrue(all((f.stream_id == 1 for f in frames[2:])))
            self.assertTrue(isinstance(frames[2], hyperframe.frame.HeadersFrame))
            self.assertTrue('END_STREAM' in frames[-1].flags)
            receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(receivedDataChunks, [b'fiver'] + dataChunks + [b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_resetAfterBody(self):
        """
        A client that immediately resets after sending the body causes Twisted
        to send no response.
        """
        frameFactory = FrameFactory()
        transport = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += buildRequestBytes(headers=self.getRequestHeaders, data=[], frameFactory=frameFactory)
        requestBytes += frameFactory.buildRstStreamFrame(streamID=1).serialize()
        a.makeConnection(transport)
        a.dataReceived(requestBytes)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertNotIn(1, a._streamCleanupCallbacks)

    def test_RequestRequiringFactorySiteInConstructor(self):
        """
        A custom L{Request} subclass that requires the site and factory in the
        constructor is able to get them.
        """
        d = defer.Deferred()

        class SuperRequest(DummyHTTPHandler):

            def __init__(self, *args, **kwargs):
                DummyHTTPHandler.__init__(self, *args, **kwargs)
                d.callback((self.channel.site, self.channel.factory))
        connection = H2Connection()
        httpFactory = http.HTTPFactory()
        connection.requestFactory = _makeRequestProxyFactory(SuperRequest)
        connection.factory = httpFactory
        connection.site = object()
        self.connectAndReceive(connection, self.getRequestHeaders, [])

        def validateFactoryAndSite(args):
            site, factory = args
            self.assertIs(site, connection.site)
            self.assertIs(factory, connection.factory)
        d.addCallback(validateFactoryAndSite)
        cleanupCallback = connection._streamCleanupCallbacks[1]
        return defer.gatherResults([d, cleanupCallback])

    def test_notifyOnCompleteRequest(self):
        """
        A request sent to a HTTP/2 connection fires the
        L{http.Request.notifyFinish} callback with a L{None} value.
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DummyHTTPHandler)
        _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def validate(result):
            self.assertIsNone(result)
        d = deferreds[0]
        d.addCallback(validate)
        cleanupCallback = connection._streamCleanupCallbacks[1]
        return defer.gatherResults([d, cleanupCallback])

    def test_notifyOnResetStream(self):
        """
        A HTTP/2 reset stream fires the L{http.Request.notifyFinish} deferred
        with L{ConnectionLost}.
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def callback(ign):
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        d = deferreds[0]
        d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildRstStreamFrame(streamID=1).serialize()
        connection.dataReceived(invalidData)
        return d

    def test_failWithProtocolError(self):
        """
        A HTTP/2 protocol error triggers the L{http.Request.notifyFinish}
        deferred for all outstanding requests with a Failure that contains the
        underlying exception.
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            self.assertIsInstance(reason, failure.Failure)
            self.assertIsInstance(reason.value, h2.exceptions.ProtocolError)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
        connection.dataReceived(invalidData)
        return defer.gatherResults(deferreds)

    def test_failOnGoaway(self):
        """
        A HTTP/2 GoAway triggers the L{http.Request.notifyFinish}
        deferred for all outstanding requests with a Failure that contains a
        RemoteGoAway error.
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildGoAwayFrame(lastStreamID=3).serialize()
        connection.dataReceived(invalidData)
        return defer.gatherResults(deferreds)

    def test_failOnStopProducing(self):
        """
        The transport telling the HTTP/2 connection to stop producing will
        fire all L{http.Request.notifyFinish} errbacks with L{error.}
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        connection.stopProducing()
        return defer.gatherResults(deferreds)

    def test_notifyOnFast400(self):
        """
        A HTTP/2 stream that has had _respondToBadRequestAndDisconnect called
        on it from a request handler calls the L{http.Request.notifyFinish}
        errback with L{ConnectionLost}.
        """
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def callback(ign):
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        d = deferreds[0]
        d.addCallbacks(callback, errback)
        stream = connection.streams[1]
        stream._respondToBadRequestAndDisconnect()
        return d

    def test_fast400WithCircuitBreaker(self):
        """
        A HTTP/2 stream that has had _respondToBadRequestAndDisconnect
        called on it does not write control frame data if its
        transport is paused and its control frame limit has been
        reached.
        """
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.requestFactory = DelayedHTTPHandler
        streamID = 1
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.dataReceived(buildRequestBytes(self.getRequestHeaders, [], frameFactory, streamID=streamID))
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        connection._respondToBadRequestAndDisconnect(streamID)
        self.assertTrue(transport.disconnected)

    def test_bufferingAutomaticFrameData(self):
        """
        If a the L{H2Connection} has been paused by the transport, it will
        not write automatic frame data triggered by writes.
        """
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        for _ in range(0, 100):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        connection.resumeProducing()
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 101)

    def test_bufferingAutomaticFrameDataWithCircuitBreaker(self):
        """
        If the L{H2Connection} has been paused by the transport, it will
        not write automatic frame data triggered by writes. If this buffer
        gets too large, the connection will be dropped.
        """
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 100
        self.assertFalse(transport.disconnecting)
        for _ in range(0, 11):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        self.assertFalse(transport.disconnecting)
        connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        self.assertTrue(transport.disconnected)

    def test_bufferingContinuesIfProducerIsPausedOnWrite(self):
        """
        If the L{H2Connection} has buffered control frames, is unpaused, and then
        paused while unbuffering, it persists the buffer and stops trying to write.
        """

        class AutoPausingStringTransport(StringTransport):

            def write(self, *args, **kwargs):
                StringTransport.write(self, *args, **kwargs)
                self.producer.pauseProducing()
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = AutoPausingStringTransport()
        transport.registerProducer(connection, True)
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        self.assertIsNotNone(connection._consumerBlocked)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertEqual(connection._bufferedControlFrameBytes, 0)
        for _ in range(0, 11):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertEqual(connection._bufferedControlFrameBytes, 9 * 11)
        connection.resumeProducing()
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)
        self.assertEqual(connection._bufferedControlFrameBytes, 9 * 10)

    def test_circuitBreakerAbortsAfterProtocolError(self):
        """
        A client that triggers a L{h2.exceptions.ProtocolError} over a
        paused connection that's reached its buffered control frame
        limit causes that connection to be aborted.
        """
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
        connection.dataReceived(invalidData)
        self.assertTrue(transport.disconnected)