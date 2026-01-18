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
class HTTP2TimeoutTests(unittest.TestCase, HTTP2TestHelpers):
    """
    The L{H2Connection} object times out idle connections.
    """
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]
    _DEFAULT = object()

    def patch_TimeoutMixin_clock(self, connection, reactor):
        """
        Unfortunately, TimeoutMixin does not allow passing an explicit reactor
        to test timeouts. For that reason, we need to monkeypatch the method
        set up by the TimeoutMixin.

        @param connection: The HTTP/2 connection object to patch.
        @type connection: L{H2Connection}

        @param reactor: The reactor whose callLater method we want.
        @type reactor: An object implementing
            L{twisted.internet.interfaces.IReactorTime}
        """
        connection.callLater = reactor.callLater

    def initiateH2Connection(self, initialData, requestFactory):
        """
        Performs test setup by building a HTTP/2 connection object, a transport
        to back it, a reactor to run in, and sending in some initial data as
        needed.

        @param initialData: The initial HTTP/2 data to be fed into the
            connection after setup.
        @type initialData: L{bytes}

        @param requestFactory: The L{Request} factory to use with the
            connection.
        """
        reactor = task.Clock()
        conn = H2Connection(reactor)
        conn.timeOut = 100
        self.patch_TimeoutMixin_clock(conn, reactor)
        transport = StringTransport()
        conn.requestFactory = _makeRequestProxyFactory(requestFactory)
        conn.makeConnection(transport)
        for byte in iterbytes(initialData):
            conn.dataReceived(byte)
        return (reactor, conn, transport)

    def assertTimedOut(self, data, frameCount, errorCode, lastStreamID):
        """
        Confirm that the data that was sent matches what we expect from a
        timeout: namely, that it ends with a GOAWAY frame carrying an
        appropriate error code and last stream ID.
        """
        frames = framesFromBytes(data)
        self.assertEqual(len(frames), frameCount)
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
        self.assertEqual(frames[-1].error_code, errorCode)
        self.assertEqual(frames[-1].last_stream_id, lastStreamID)

    def prepareAbortTest(self, abortTimeout=_DEFAULT):
        """
        Does the common setup for tests that want to test the aborting
        functionality of the HTTP/2 stack.

        @param abortTimeout: The value to use for the abortTimeout. Defaults to
            whatever is set on L{H2Connection.abortTimeout}.
        @type abortTimeout: L{int} or L{None}

        @return: A tuple of the reactor being used for the connection, the
            connection itself, and the transport.
        """
        if abortTimeout is self._DEFAULT:
            abortTimeout = H2Connection.abortTimeout
        frameFactory = FrameFactory()
        initialData = frameFactory.clientConnectionPreface()
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        conn.abortTimeout = abortTimeout
        reactor.advance(100)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        return (reactor, conn, transport)

    def test_timeoutAfterInactivity(self):
        """
        When a L{H2Connection} does not receive any data for more than the
        time out interval, it closes the connection cleanly.
        """
        frameFactory = FrameFactory()
        initialData = frameFactory.clientConnectionPreface()
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        preamble = transport.value()
        reactor.advance(99)
        self.assertEqual(preamble, transport.value())
        self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)

    def test_timeoutResetByRequestData(self):
        """
        When a L{H2Connection} receives data, the timeout is reset.
        """
        frameFactory = FrameFactory()
        initialData = b''
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        for byte in iterbytes(frameFactory.clientConnectionPreface()):
            conn.dataReceived(byte)
            reactor.advance(99)
            self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)

    def test_timeoutResetByResponseData(self):
        """
        When a L{H2Connection} sends data, the timeout is reset.
        """
        frameFactory = FrameFactory()
        initialData = b''
        requests = []
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))

        def saveRequest(stream, queued):
            req = DelayedHTTPHandler(stream, queued=queued)
            requests.append(req)
            return req
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=saveRequest)
        conn.dataReceived(frameFactory.clientConnectionPreface())
        reactor.advance(99)
        self.assertEquals(len(requests), 1)
        for x in range(10):
            requests[0].write(b'some bytes')
            reactor.advance(99)
            self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=13, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)

    def test_timeoutWithProtocolErrorIfStreamsOpen(self):
        """
        When a L{H2Connection} times out with active streams, the error code
        returned is L{h2.errors.ErrorCodes.PROTOCOL_ERROR}.
        """
        frameFactory = FrameFactory()
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
        reactor.advance(101)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)
        self.assertTrue(transport.disconnecting)

    def test_noTimeoutIfConnectionLost(self):
        """
        When a L{H2Connection} loses its connection it cancels its timeout.
        """
        frameFactory = FrameFactory()
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))
        reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
        sentData = transport.value()
        oldCallCount = len(reactor.getDelayedCalls())
        conn.connectionLost('reason')
        currentCallCount = len(reactor.getDelayedCalls())
        self.assertEqual(oldCallCount - 1, currentCallCount)
        reactor.advance(101)
        self.assertEqual(transport.value(), sentData)

    def test_timeoutEventuallyForcesConnectionClosed(self):
        """
        When a L{H2Connection} has timed the connection out, and the transport
        doesn't get torn down within 15 seconds, it gets forcibly closed.
        """
        reactor, conn, transport = self.prepareAbortTest()
        reactor.advance(14)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        reactor.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)

    def test_losingConnectionCancelsTheAbort(self):
        """
        When a L{H2Connection} has timed the connection out, getting
        C{connectionLost} called on it cancels the forcible connection close.
        """
        reactor, conn, transport = self.prepareAbortTest()
        reactor.advance(14)
        conn.connectionLost(None)
        reactor.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_losingConnectionWithNoAbortTimeOut(self):
        """
        When a L{H2Connection} has timed the connection out but the
        C{abortTimeout} is set to L{None}, the connection is never aborted.
        """
        reactor, conn, transport = self.prepareAbortTest(abortTimeout=None)
        reactor.advance(2 ** 32)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_connectionLostAfterForceClose(self):
        """
        If a timed out transport doesn't close after 15 seconds, the
        L{HTTPChannel} will forcibly close it.
        """
        reactor, conn, transport = self.prepareAbortTest()
        reactor.advance(15)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)
        conn.connectionLost(error.ConnectionDone)

    def test_timeOutClientThatSendsOnlyInvalidFrames(self):
        """
        A client that sends only invalid frames is eventually timed out.
        """
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.timeOut = 60
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        for _ in range(connection.timeOut + connection.abortTimeout):
            connection.dataReceived(frameFactory.buildRstStreamFrame(1).serialize())
            memoryReactor.advance(1)
        self.assertTrue(transport.disconnected)