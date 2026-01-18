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
class H2FlowControlTests(unittest.TestCase, HTTP2TestHelpers):
    """
    Tests that ensure that we handle HTTP/2 flow control limits appropriately.
    """
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code')]
    getResponseData = b"'''\nNone\n'''\n"
    postRequestHeaders = [(b':method', b'POST'), (b':authority', b'localhost'), (b':path', b'/post_endpoint'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'content-length', b'25')]
    postRequestData = [b'hello ', b'world, ', b"it's ", b'http/2!']
    postResponseData = b"'''\n25\nhello world, it's http/2!'''\n"

    def test_bufferExcessData(self):
        """
        When a L{Request} object is not using C{IProducer} to generate data and
        so is not having backpressure exerted on it, the L{H2Stream} object
        will buffer data until the flow control window is opened.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        bonusFrames = len(self.getResponseData) - 5
        for _ in range(bonusFrames):
            frame = f.buildWindowUpdateFrame(streamID=1, increment=1)
            a.dataReceived(frame.serialize())

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
            self.assertEqual(self.getResponseData, actualResponseData)
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_producerBlockingUnblocking(self):
        """
        L{Request} objects that have registered producers get blocked and
        unblocked according to HTTP/2 flow control.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=0, increment=5).serialize())
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume'])
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume', 'pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume', 'pause', 'resume'])
        request.unregisterProducer()
        request.finish()

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'helloworld', b'helloworld', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_flowControlExact(self):
        """
        Exactly filling the flow control window still blocks producers.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        request.write(b'h')

        def window_open():
            a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
            self.assertTrue(stream._producerProducing)
            self.assertEqual(request.producer.events, ['pause', 'resume'])
            request.unregisterProducer()
            request.finish()
        windowDefer = task.deferLater(reactor, 0, window_open)

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'hello', b'world', b'h', b''])
        validateDefer = a._streamCleanupCallbacks[1].addCallback(validate)
        return defer.DeferredList([windowDefer, validateDefer])

    def test_endingBlockedStream(self):
        """
        L{Request} objects that end a stream that is currently blocked behind
        flow control can still end the stream and get cleaned up.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)
        reactor.callLater(0, a.dataReceived, f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'hello', b'world', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_responseWithoutBody(self):
        """
        We safely handle responses without bodies.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b''])
        return cleanupCallback.addCallback(validate)

    def test_windowUpdateForCompleteStream(self):
        """
        WindowUpdate frames received after we've completed the stream are
        safely handled.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b''])
        return cleanupCallback.addCallback(validate)

    def test_producerUnblocked(self):
        """
        L{Request} objects that have registered producers that are not blocked
        behind flow control do not have their producer notified.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'word')
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, [])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, [])
        request.unregisterProducer()
        request.finish()

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'word', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_unnecessaryWindowUpdate(self):
        """
        When a WindowUpdate frame is received for the whole connection but no
        data is currently waiting, nothing exciting happens.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        frames.insert(1, f.buildWindowUpdateFrame(streamID=0, increment=5))
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(streamID):
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
            self.assertEqual(self.postResponseData, actualResponseData)
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_unnecessaryWindowUpdateForStream(self):
        """
        When a WindowUpdate frame is received for a stream but no data is
        currently waiting, that stream is not marked as unblocked and the
        priority tree continues to assert that no stream can progress.
        """
        f = FrameFactory()
        transport = StringTransport()
        conn = H2Connection()
        conn.requestFactory = DummyHTTPHandlerProxy
        frames = []
        frames.append(f.buildHeadersFrame(headers=self.postRequestHeaders, streamID=1))
        frames.append(f.buildWindowUpdateFrame(streamID=1, increment=5))
        data = f.clientConnectionPreface()
        data += b''.join((f.serialize() for f in frames))
        conn.makeConnection(transport)
        conn.dataReceived(data)
        self.assertAllStreamsBlocked(conn)

    def test_windowUpdateAfterTerminate(self):
        """
        When a WindowUpdate frame is received for a stream that has been
        aborted it is ignored.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.streams[1].abortConnection()
        windowUpdateFrame = f.buildWindowUpdateFrame(streamID=1, increment=5)
        a.dataReceived(windowUpdateFrame.serialize())
        frames = framesFromBytes(b.value())
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.RstStreamFrame))

    def test_windowUpdateAfterComplete(self):
        """
        When a WindowUpdate frame is received for a stream that has been
        completed it is ignored.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def update_window(*args):
            windowUpdateFrame = f.buildWindowUpdateFrame(streamID=1, increment=5)
            a.dataReceived(windowUpdateFrame.serialize())

        def validate(*args):
            frames = framesFromBytes(b.value())
            self.assertIn('END_STREAM', frames[-1].flags)
        d = a._streamCleanupCallbacks[1].addCallback(update_window)
        return d.addCallback(validate)

    def test_dataAndRstStream(self):
        """
        When a DATA frame is received at the same time as RST_STREAM,
        Twisted does not send WINDOW_UPDATE frames for the stream.
        """
        frameFactory = FrameFactory()
        transport = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frameData = [b'\x00' * 2 ** 14] * 4
        bodyLength = f'{sum((len(data) for data in frameData))}'
        headers = self.postRequestHeaders[:-1] + [('content-length', bodyLength)]
        frames = buildRequestFrames(headers=headers, data=frameData, frameFactory=frameFactory)
        del frames[-1]
        frames.append(frameFactory.buildRstStreamFrame(streamID=1, errorCode=h2.errors.ErrorCodes.INTERNAL_ERROR))
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(transport)
        a.dataReceived(requestBytes)
        frames = framesFromBytes(transport.value())
        windowUpdateFrameIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.WindowUpdateFrame)]
        self.assertEqual([0], windowUpdateFrameIDs)
        headersFrames = [f for f in frames if isinstance(f, hyperframe.frame.HeadersFrame)]
        dataFrames = [f for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertFalse(headersFrames)
        self.assertFalse(dataFrames)

    def test_abortRequestWithCircuitBreaker(self):
        """
        Aborting a request associated with a paused connection that's
        reached its buffered control frame limit causes that
        connection to be aborted.
        """
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        streamID = 1
        headersFrameData = frameFactory.buildHeadersFrame(headers=self.postRequestHeaders, streamID=streamID).serialize()
        connection.dataReceived(headersFrameData)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        transport.clear()
        connection.abortRequest(streamID)
        self.assertFalse(transport.value())
        self.assertTrue(transport.disconnected)