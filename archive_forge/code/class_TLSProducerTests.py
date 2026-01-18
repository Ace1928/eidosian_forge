from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
class TLSProducerTests(TestCase):
    """
    The TLS transport must support the IConsumer interface.
    """

    def drain(self, transport, allowEmpty=False):
        """
        Drain the bytes currently pending write from a L{StringTransport}, then
        clear it, since those bytes have been consumed.

        @param transport: The L{StringTransport} to get the bytes from.
        @type transport: L{StringTransport}

        @param allowEmpty: Allow the test to pass even if the transport has no
            outgoing bytes in it.
        @type allowEmpty: L{bool}

        @return: the outgoing bytes from the given transport
        @rtype: L{bytes}
        """
        value = transport.value()
        transport.clear()
        self.assertEqual(bool(allowEmpty or value), True)
        return value

    def setupStreamingProducer(self, transport=None, fakeConnection=None, server=False, serverMethod=None):
        """
        Create a new client-side protocol that is connected to a remote TLS server.

        @param serverMethod: The TLS method accepted by the server-side. Set to to C{None} to use the default method used by your OpenSSL library.

        @return: A tuple with high level client protocol, the low-level client-side TLS protocol, and a producer that is used to send data to the client.
        """

        class HistoryStringTransport(StringTransport):

            def __init__(self):
                StringTransport.__init__(self)
                self.producerHistory = []

            def pauseProducing(self):
                self.producerHistory.append('pause')
                StringTransport.pauseProducing(self)

            def resumeProducing(self):
                self.producerHistory.append('resume')
                StringTransport.resumeProducing(self)

            def stopProducing(self):
                self.producerHistory.append('stop')
                StringTransport.stopProducing(self)
        applicationProtocol, tlsProtocol = buildTLSProtocol(transport=transport, fakeConnection=fakeConnection, server=server, serverMethod=serverMethod)
        producer = HistoryStringTransport()
        applicationProtocol.transport.registerProducer(producer, True)
        self.assertTrue(tlsProtocol.transport.streaming)
        return (applicationProtocol, tlsProtocol, producer)

    def flushTwoTLSProtocols(self, tlsProtocol, serverTLSProtocol):
        """
        Transfer bytes back and forth between two TLS protocols.
        """
        for i in range(3):
            clientData = self.drain(tlsProtocol.transport, True)
            if clientData:
                serverTLSProtocol.dataReceived(clientData)
            serverData = self.drain(serverTLSProtocol.transport, True)
            if serverData:
                tlsProtocol.dataReceived(serverData)
            if not serverData and (not clientData):
                break
        self.assertEqual(tlsProtocol.transport.value(), b'')
        self.assertEqual(serverTLSProtocol.transport.value(), b'')

    def test_streamingProducerPausedInNormalMode(self):
        """
        When the TLS transport is not blocked on reads, it correctly calls
        pauseProducing on the registered producer.
        """
        _, tlsProtocol, producer = self.setupStreamingProducer()
        tlsProtocol.transport.producer.pauseProducing()
        self.assertEqual(producer.producerState, 'paused')
        self.assertEqual(producer.producerHistory, ['pause'])
        self.assertTrue(tlsProtocol._producer._producerPaused)

    def test_streamingProducerResumedInNormalMode(self):
        """
        When the TLS transport is not blocked on reads, it correctly calls
        resumeProducing on the registered producer.
        """
        _, tlsProtocol, producer = self.setupStreamingProducer()
        tlsProtocol.transport.producer.pauseProducing()
        self.assertEqual(producer.producerHistory, ['pause'])
        tlsProtocol.transport.producer.resumeProducing()
        self.assertEqual(producer.producerState, 'producing')
        self.assertEqual(producer.producerHistory, ['pause', 'resume'])
        self.assertFalse(tlsProtocol._producer._producerPaused)

    def test_streamingProducerPausedInWriteBlockedOnReadMode(self):
        """
        When the TLS transport is blocked on reads, it correctly calls
        pauseProducing on the registered producer.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        clientProtocol.transport.write(b'hello')
        tlsProtocol.factory._clock.advance(0)
        self.assertEqual(producer.producerState, 'paused')
        self.assertEqual(producer.producerHistory, ['pause'])
        self.assertTrue(tlsProtocol._producer._producerPaused)

    def test_streamingProducerResumedInWriteBlockedOnReadMode(self):
        """
        When the TLS transport is blocked on reads, it correctly calls
        resumeProducing on the registered producer.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        clientProtocol.transport.write(b'hello world' * 320000)
        self.assertEqual(producer.producerHistory, ['pause'])
        serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
        self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
        self.assertEqual(producer.producerHistory, ['pause', 'resume'])
        self.assertFalse(tlsProtocol._producer._producerPaused)
        self.assertFalse(tlsProtocol.transport.disconnecting)
        self.assertEqual(producer.producerState, 'producing')

    def test_streamingProducerTwice(self):
        """
        Registering a streaming producer twice throws an exception.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        originalProducer = tlsProtocol._producer
        producer2 = object()
        self.assertRaises(RuntimeError, clientProtocol.transport.registerProducer, producer2, True)
        self.assertIs(tlsProtocol._producer, originalProducer)

    def test_streamingProducerUnregister(self):
        """
        Unregistering a streaming producer removes it, reverting to initial state.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        clientProtocol.transport.unregisterProducer()
        self.assertIsNone(tlsProtocol._producer)
        self.assertIsNone(tlsProtocol.transport.producer)

    def test_streamingProducerUnregisterTwice(self):
        """
        Unregistering a streaming producer when no producer is registered is
        safe.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        clientProtocol.transport.unregisterProducer()
        clientProtocol.transport.unregisterProducer()
        self.assertIsNone(tlsProtocol._producer)
        self.assertIsNone(tlsProtocol.transport.producer)

    def loseConnectionWithProducer(self, writeBlockedOnRead):
        """
        Common code for tests involving writes by producer after
        loseConnection is called.
        """
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
        serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
        if not writeBlockedOnRead:
            self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
        else:
            pass
        clientProtocol.transport.write(b'x ')
        clientProtocol.transport.loseConnection()
        self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
        self.assertFalse(tlsProtocol.transport.disconnecting)
        self.assertFalse('stop' in producer.producerHistory)
        clientProtocol.transport.write(b'hello')
        clientProtocol.transport.writeSequence([b' ', b'world'])
        tlsProtocol.factory._clock.advance(0)
        clientProtocol.transport.unregisterProducer()
        self.assertNotEqual(tlsProtocol.transport.value(), b'')
        self.assertFalse(tlsProtocol.transport.disconnecting)
        clientProtocol.transport.write(b"won't")
        clientProtocol.transport.writeSequence([b"won't!"])
        tlsProtocol.factory._clock.advance(0)
        self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
        self.assertTrue(tlsProtocol.transport.disconnecting)
        self.assertEqual(b''.join(serverProtocol.received), b'x hello world')

    def test_streamingProducerLoseConnectionWithProducer(self):
        """
        loseConnection() waits for the producer to unregister itself, then
        does a clean TLS close alert, then closes the underlying connection.
        """
        return self.loseConnectionWithProducer(False)

    def test_streamingProducerLoseConnectionWithProducerWBOR(self):
        """
        Even when writes are blocked on reading, loseConnection() waits for
        the producer to unregister itself, then does a clean TLS close alert,
        then closes the underlying connection.
        """
        return self.loseConnectionWithProducer(True)

    def test_streamingProducerBothTransportsDecideToPause(self):
        """
        pauseProducing() events can come from both the TLS transport layer and
        the underlying transport. In this case, both decide to pause,
        underlying first.
        """

        class PausingStringTransport(StringTransport):
            _didPause = False

            def write(self, data):
                if not self._didPause and self.producer is not None:
                    self._didPause = True
                    self.producer.pauseProducing()
                StringTransport.write(self, data)

        class TLSConnection:

            def __init__(self):
                self.l = []

            def send(self, data):
                if not self.l:
                    data = data[:-1]
                if len(self.l) == 1:
                    self.l.append('paused')
                    raise WantReadError()
                self.l.append(data)
                return len(data)

            def set_connect_state(self):
                pass

            def do_handshake(self):
                pass

            def bio_write(self, data):
                pass

            def bio_read(self, size):
                return b'X'

            def recv(self, size):
                raise WantReadError()
        transport = PausingStringTransport()
        clientProtocol, tlsProtocol, producer = self.setupStreamingProducer(transport, fakeConnection=TLSConnection())
        self.assertEqual(producer.producerState, 'producing')
        clientProtocol.transport.write(b'hello')
        tlsProtocol.factory._clock.advance(0)
        self.assertEqual(producer.producerState, 'paused')
        self.assertEqual(producer.producerHistory, ['pause'])
        tlsProtocol.transport.producer.resumeProducing()
        self.assertEqual(producer.producerState, 'producing')
        self.assertEqual(producer.producerHistory, ['pause', 'resume'])
        tlsProtocol.dataReceived(b'hello')
        self.assertEqual(producer.producerState, 'producing')
        self.assertEqual(producer.producerHistory, ['pause', 'resume'])

    def test_streamingProducerStopProducing(self):
        """
        If the underlying transport tells its producer to stopProducing(),
        this is passed on to the high-level producer.
        """
        _, tlsProtocol, producer = self.setupStreamingProducer()
        tlsProtocol.transport.producer.stopProducing()
        self.assertEqual(producer.producerState, 'stopped')

    def test_nonStreamingProducer(self):
        """
        Non-streaming producers get wrapped as streaming producers.
        """
        clientProtocol, tlsProtocol = buildTLSProtocol()
        producer = NonStreamingProducer(clientProtocol.transport)
        clientProtocol.transport.registerProducer(producer, False)
        streamingProducer = tlsProtocol.transport.producer._producer
        self.assertIsInstance(streamingProducer, _PullToPush)
        self.assertEqual(streamingProducer._producer, producer)
        self.assertEqual(streamingProducer._consumer, clientProtocol.transport)
        self.assertTrue(tlsProtocol.transport.streaming)

        def done(ignore):
            self.assertIsNone(producer.consumer)
            self.assertIsNone(tlsProtocol.transport.producer)
            self.assertTrue(streamingProducer._finished)
        producer.result.addCallback(done)
        serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
        self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
        return producer.result

    def test_interface(self):
        """
        L{_ProducerMembrane} implements L{IPushProducer}.
        """
        producer = StringTransport()
        membrane = _ProducerMembrane(producer)
        self.assertTrue(verifyObject(IPushProducer, membrane))

    def registerProducerAfterConnectionLost(self, streaming):
        """
        If a producer is registered after the transport has disconnected, the
        producer is not used, and its stopProducing method is called.
        """
        clientProtocol, tlsProtocol = buildTLSProtocol()
        clientProtocol.connectionLost = lambda reason: reason.trap(Error, ConnectionLost)

        class Producer:
            stopped = False

            def resumeProducing(self):
                return 1 / 0

            def stopProducing(self):
                self.stopped = True
        tlsProtocol.connectionLost(Failure(ConnectionDone()))
        producer = Producer()
        tlsProtocol.registerProducer(producer, False)
        self.assertIsNone(tlsProtocol.transport.producer)
        self.assertTrue(producer.stopped)

    def test_streamingProducerAfterConnectionLost(self):
        """
        If a streaming producer is registered after the transport has
        disconnected, the producer is not used, and its stopProducing method
        is called.
        """
        self.registerProducerAfterConnectionLost(True)

    def test_nonStreamingProducerAfterConnectionLost(self):
        """
        If a non-streaming producer is registered after the transport has
        disconnected, the producer is not used, and its stopProducing method
        is called.
        """
        self.registerProducerAfterConnectionLost(False)