import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class ChannelProductionTests(unittest.TestCase):
    """
    Tests for the way HTTPChannel manages backpressure.
    """
    request = b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n'

    def buildChannelAndTransport(self, transport, requestFactory):
        """
        Setup a L{HTTPChannel} and a transport and associate them.

        @param transport: A transport to back the L{HTTPChannel}
        @param requestFactory: An object that can construct L{Request} objects.
        @return: A tuple of the channel and the transport.
        """
        transport = transport
        channel = http.HTTPChannel()
        channel.requestFactory = _makeRequestProxyFactory(requestFactory)
        channel.makeConnection(transport)
        return (channel, transport)

    def test_HTTPChannelIsAProducer(self):
        """
        L{HTTPChannel} registers itself as a producer with its transport when a
        connection is made.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        self.assertEqual(transport.producer, channel)
        self.assertTrue(transport.streaming)

    def test_HTTPChannelUnregistersSelfWhenCallingLoseConnection(self):
        """
        L{HTTPChannel} unregisters itself when it has loseConnection called.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        channel.loseConnection()
        self.assertIs(transport.producer, None)
        self.assertIs(transport.streaming, None)

    def test_HTTPChannelRejectsMultipleProducers(self):
        """
        If two producers are registered on a L{HTTPChannel} without the first
        being unregistered, a L{RuntimeError} is thrown.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        channel.registerProducer(DummyProducer(), True)
        self.assertRaises(RuntimeError, channel.registerProducer, DummyProducer(), True)

    def test_HTTPChannelCanUnregisterWithNoProducer(self):
        """
        If there is no producer, the L{HTTPChannel} can still have
        C{unregisterProducer} called.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        channel.unregisterProducer()
        self.assertIs(channel._requestProducer, None)

    def test_HTTPChannelStopWithNoRequestOutstanding(self):
        """
        If there is no request producer currently registered, C{stopProducing}
        does nothing.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        channel.unregisterProducer()
        self.assertIs(channel._requestProducer, None)

    def test_HTTPChannelStopRequestProducer(self):
        """
        If there is a request producer registered with L{HTTPChannel}, calling
        C{stopProducing} causes that producer to be stopped as well.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DelayedHTTPHandler)
        channel.dataReceived(self.request)
        request = channel.requests[0].original
        producer = DummyProducer()
        request.registerProducer(producer, True)
        self.assertEqual(producer.events, [])
        channel.stopProducing()
        self.assertEqual(producer.events, ['stop'])

    def test_HTTPChannelPropagatesProducingFromTransportToTransport(self):
        """
        When L{HTTPChannel} has C{pauseProducing} called on it by the transport
        it will call C{pauseProducing} on the transport. When unpaused, the
        L{HTTPChannel} will call C{resumeProducing} on its transport.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyHTTPHandler)
        self.assertEqual(transport.producerState, 'producing')
        channel.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        channel.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_HTTPChannelPropagatesPausedProductionToRequest(self):
        """
        If a L{Request} object has registered itself as a producer with a
        L{HTTPChannel} object, and the L{HTTPChannel} object is paused, both
        the transport and L{Request} objects get paused.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DelayedHTTPHandler)
        channel._optimisticEagerReadSize = 0
        channel.dataReceived(self.request)
        channel.dataReceived(b'123')
        request = channel.requests[0].original
        producer = DummyProducer()
        request.registerProducer(producer, True)
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(producer.events, [])
        channel.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(producer.events, ['pause'])
        channel.resumeProducing()
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(producer.events, ['pause', 'resume'])
        request.unregisterProducer()
        request.delayedProcess()
        self.assertEqual(transport.producerState, 'producing')

    def test_HTTPChannelStaysPausedWhenRequestCompletes(self):
        """
        If a L{Request} object completes its response while the transport is
        paused, the L{HTTPChannel} does not resume the transport.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DelayedHTTPHandler)
        channel._optimisticEagerReadSize = 0
        channel.dataReceived(self.request)
        channel.dataReceived(b'extra')
        request = channel.requests[0].original
        producer = DummyProducer()
        request.registerProducer(producer, True)
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(producer.events, [])
        channel.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(producer.events, ['pause'])
        request.unregisterProducer()
        request.delayedProcess()
        self.assertEqual(transport.producerState, 'paused')
        channel.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_HTTPChannelToleratesDataWhenTransportPaused(self):
        """
        If the L{HTTPChannel} has paused the transport, it still tolerates
        receiving data, and does not attempt to pause the transport again.
        """

        class NoDoublePauseTransport(StringTransport):
            """
            A version of L{StringTransport} that fails tests if it is paused
            while already paused.
            """

            def pauseProducing(self):
                if self.producerState == 'paused':
                    raise RuntimeError('Transport was paused twice!')
                StringTransport.pauseProducing(self)
        transport = NoDoublePauseTransport()
        transport.pauseProducing()
        self.assertRaises(RuntimeError, transport.pauseProducing)
        channel, transport = self.buildChannelAndTransport(NoDoublePauseTransport(), DummyHTTPHandler)
        self.assertEqual(transport.producerState, 'producing')
        channel.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        channel.dataReceived(self.request)
        self.assertEqual(transport.producerState, 'paused')
        self.assertTrue(transport.value().startswith(b'HTTP/1.1 200 OK\r\n'))
        channel.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_HTTPChannelToleratesPullProducers(self):
        """
        If the L{HTTPChannel} has a L{IPullProducer} registered with it it can
        adapt that producer into an L{IPushProducer}.
        """
        channel, transport = self.buildChannelAndTransport(StringTransport(), DummyPullProducerHandler)
        transport = StringTransport()
        channel = http.HTTPChannel()
        channel.requestFactory = DummyPullProducerHandlerProxy
        channel.makeConnection(transport)
        channel.dataReceived(self.request)
        request = channel.requests[0].original
        responseComplete = request._actualProducer.result

        def validate(ign):
            responseBody = transport.value().split(b'\r\n\r\n', 1)[1]
            expectedResponseBody = b'1\r\n0\r\n1\r\n1\r\n1\r\n2\r\n1\r\n3\r\n1\r\n4\r\n1\r\n5\r\n1\r\n6\r\n1\r\n7\r\n1\r\n8\r\n1\r\n9\r\n'
            self.assertEqual(responseBody, expectedResponseBody)
        return responseComplete.addCallback(validate)

    def test_HTTPChannelUnregistersSelfWhenTimingOut(self):
        """
        L{HTTPChannel} unregisters itself when it times out a connection.
        """
        clock = Clock()
        transport = StringTransport()
        channel = http.HTTPChannel()
        channel.timeOut = 100
        channel.callLater = clock.callLater
        channel.makeConnection(transport)
        clock.advance(99)
        self.assertIs(transport.producer, channel)
        self.assertIs(transport.streaming, True)
        clock.advance(1)
        self.assertIs(transport.producer, None)
        self.assertIs(transport.streaming, None)