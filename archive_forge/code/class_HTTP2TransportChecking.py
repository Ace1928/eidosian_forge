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
class HTTP2TransportChecking(unittest.TestCase, HTTP2TestHelpers):
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]

    def test_registerProducerWithTransport(self):
        """
        L{H2Connection} can be registered with the transport as a producer.
        """
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        b.registerProducer(a, True)
        self.assertTrue(b.producer is a)

    def test_pausingProducerPreventsDataSend(self):
        """
        L{H2Connection} can be paused by its consumer. When paused it stops
        sending data to the transport.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.getRequestHeaders, [], f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        b.registerProducer(a, True)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.pauseProducing()
        cleanupCallback = a._streamCleanupCallbacks[1]

        def validateNotSent(*args):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 2)
            self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            return cleanupCallback

        def validateComplete(*args):
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        d = task.deferLater(reactor, 0.01, validateNotSent)
        d.addCallback(validateComplete)
        return d

    def test_stopProducing(self):
        """
        L{H2Connection} can be stopped by its producer. That causes it to lose
        its transport.
        """
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.getRequestHeaders, [], f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        b.registerProducer(a, True)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.stopProducing()
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 2)
        self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
        self.assertFalse(a._stillProducing)

    def test_passthroughHostAndPeer(self):
        """
        A L{H2Stream} object correctly passes through host and peer information
        from its L{H2Connection}.
        """
        hostAddress = IPv4Address('TCP', '17.52.24.8', 443)
        peerAddress = IPv4Address('TCP', '17.188.0.12', 32008)
        frameFactory = FrameFactory()
        transport = StringTransport(hostAddress=hostAddress, peerAddress=peerAddress)
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        connection.makeConnection(transport)
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += b''.join((frame.serialize() for frame in frames))
        for byte in iterbytes(requestBytes):
            connection.dataReceived(byte)
        stream = connection.streams[1]
        self.assertEqual(stream.getHost(), hostAddress)
        self.assertEqual(stream.getPeer(), peerAddress)
        cleanupCallback = connection._streamCleanupCallbacks[1]

        def validate(*args):
            self.assertEqual(stream.getHost(), hostAddress)
            self.assertEqual(stream.getPeer(), peerAddress)
        return cleanupCallback.addCallback(validate)