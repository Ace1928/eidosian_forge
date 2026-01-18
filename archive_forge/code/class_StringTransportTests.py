from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
class StringTransportTests(TestCase):
    """
    Tests for L{twisted.internet.testing.StringTransport}.
    """

    def setUp(self) -> None:
        self.transport = StringTransport()

    def test_interfaces(self) -> None:
        """
        L{StringTransport} instances provide L{ITransport}, L{IPushProducer},
        and L{IConsumer}.
        """
        self.assertTrue(verifyObject(ITransport, self.transport))
        self.assertTrue(verifyObject(IPushProducer, self.transport))
        self.assertTrue(verifyObject(IConsumer, self.transport))

    def test_registerProducer(self) -> None:
        """
        L{StringTransport.registerProducer} records the arguments supplied to
        it as instance attributes.
        """
        producer = object()
        streaming = object()
        self.transport.registerProducer(producer, streaming)
        self.assertIs(self.transport.producer, producer)
        self.assertIs(self.transport.streaming, streaming)

    def test_disallowedRegisterProducer(self) -> None:
        """
        L{StringTransport.registerProducer} raises L{RuntimeError} if a
        producer is already registered.
        """
        producer = object()
        self.transport.registerProducer(producer, True)
        self.assertRaises(RuntimeError, self.transport.registerProducer, object(), False)
        self.assertIs(self.transport.producer, producer)
        self.assertTrue(self.transport.streaming)

    def test_unregisterProducer(self) -> None:
        """
        L{StringTransport.unregisterProducer} causes the transport to forget
        about the registered producer and makes it possible to register a new
        one.
        """
        oldProducer = object()
        newProducer = object()
        self.transport.registerProducer(oldProducer, False)
        self.transport.unregisterProducer()
        self.assertIsNone(self.transport.producer)
        self.transport.registerProducer(newProducer, True)
        self.assertIs(self.transport.producer, newProducer)
        self.assertTrue(self.transport.streaming)

    def test_invalidUnregisterProducer(self) -> None:
        """
        L{StringTransport.unregisterProducer} raises L{RuntimeError} if called
        when no producer is registered.
        """
        self.assertRaises(RuntimeError, self.transport.unregisterProducer)

    def test_initialProducerState(self) -> None:
        """
        L{StringTransport.producerState} is initially C{'producing'}.
        """
        self.assertEqual(self.transport.producerState, 'producing')

    def test_pauseProducing(self) -> None:
        """
        L{StringTransport.pauseProducing} changes the C{producerState} of the
        transport to C{'paused'}.
        """
        self.transport.pauseProducing()
        self.assertEqual(self.transport.producerState, 'paused')

    def test_resumeProducing(self) -> None:
        """
        L{StringTransport.resumeProducing} changes the C{producerState} of the
        transport to C{'producing'}.
        """
        self.transport.pauseProducing()
        self.transport.resumeProducing()
        self.assertEqual(self.transport.producerState, 'producing')

    def test_stopProducing(self) -> None:
        """
        L{StringTransport.stopProducing} changes the C{'producerState'} of the
        transport to C{'stopped'}.
        """
        self.transport.stopProducing()
        self.assertEqual(self.transport.producerState, 'stopped')

    def test_stoppedTransportCannotPause(self) -> None:
        """
        L{StringTransport.pauseProducing} raises L{RuntimeError} if the
        transport has been stopped.
        """
        self.transport.stopProducing()
        self.assertRaises(RuntimeError, self.transport.pauseProducing)

    def test_stoppedTransportCannotResume(self) -> None:
        """
        L{StringTransport.resumeProducing} raises L{RuntimeError} if the
        transport has been stopped.
        """
        self.transport.stopProducing()
        self.assertRaises(RuntimeError, self.transport.resumeProducing)

    def test_disconnectingTransportCannotPause(self) -> None:
        """
        L{StringTransport.pauseProducing} raises L{RuntimeError} if the
        transport is being disconnected.
        """
        self.transport.loseConnection()
        self.assertRaises(RuntimeError, self.transport.pauseProducing)

    def test_disconnectingTransportCannotResume(self) -> None:
        """
        L{StringTransport.resumeProducing} raises L{RuntimeError} if the
        transport is being disconnected.
        """
        self.transport.loseConnection()
        self.assertRaises(RuntimeError, self.transport.resumeProducing)

    def test_loseConnectionSetsDisconnecting(self) -> None:
        """
        L{StringTransport.loseConnection} toggles the C{disconnecting} instance
        variable to C{True}.
        """
        self.assertFalse(self.transport.disconnecting)
        self.transport.loseConnection()
        self.assertTrue(self.transport.disconnecting)

    def test_specifiedHostAddress(self) -> None:
        """
        If a host address is passed to L{StringTransport.__init__}, that
        value is returned from L{StringTransport.getHost}.
        """
        address = object()
        self.assertIs(StringTransport(address).getHost(), address)

    def test_specifiedPeerAddress(self) -> None:
        """
        If a peer address is passed to L{StringTransport.__init__}, that
        value is returned from L{StringTransport.getPeer}.
        """
        address = object()
        self.assertIs(StringTransport(peerAddress=address).getPeer(), address)

    def test_defaultHostAddress(self) -> None:
        """
        If no host address is passed to L{StringTransport.__init__}, an
        L{IPv4Address} is returned from L{StringTransport.getHost}.
        """
        address = StringTransport().getHost()
        self.assertIsInstance(address, IPv4Address)

    def test_defaultPeerAddress(self) -> None:
        """
        If no peer address is passed to L{StringTransport.__init__}, an
        L{IPv4Address} is returned from L{StringTransport.getPeer}.
        """
        address = StringTransport().getPeer()
        self.assertIsInstance(address, IPv4Address)