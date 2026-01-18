from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
class StrictPushProducerTests(TestCase):
    """
    Tests for L{StrictPushProducer}.
    """

    def _initial(self) -> StrictPushProducer:
        """
        @return: A new L{StrictPushProducer} which has not been through any state
            changes.
        """
        return StrictPushProducer()

    def _stopped(self) -> StrictPushProducer:
        """
        @return: A new, stopped L{StrictPushProducer}.
        """
        producer = StrictPushProducer()
        producer.stopProducing()
        return producer

    def _paused(self) -> StrictPushProducer:
        """
        @return: A new, paused L{StrictPushProducer}.
        """
        producer = StrictPushProducer()
        producer.pauseProducing()
        return producer

    def _resumed(self) -> StrictPushProducer:
        """
        @return: A new L{StrictPushProducer} which has been paused and resumed.
        """
        producer = StrictPushProducer()
        producer.pauseProducing()
        producer.resumeProducing()
        return producer

    def assertStopped(self, producer: StrictPushProducer) -> None:
        """
        Assert that the given producer is in the stopped state.

        @param producer: The producer to verify.
        @type producer: L{StrictPushProducer}
        """
        self.assertEqual(producer._state, 'stopped')

    def assertPaused(self, producer: StrictPushProducer) -> None:
        """
        Assert that the given producer is in the paused state.

        @param producer: The producer to verify.
        @type producer: L{StrictPushProducer}
        """
        self.assertEqual(producer._state, 'paused')

    def assertRunning(self, producer: StrictPushProducer) -> None:
        """
        Assert that the given producer is in the running state.

        @param producer: The producer to verify.
        @type producer: L{StrictPushProducer}
        """
        self.assertEqual(producer._state, 'running')

    def test_stopThenStop(self) -> None:
        """
        L{StrictPushProducer.stopProducing} raises L{ValueError} if called when
        the producer is stopped.
        """
        self.assertRaises(ValueError, self._stopped().stopProducing)

    def test_stopThenPause(self) -> None:
        """
        L{StrictPushProducer.pauseProducing} raises L{ValueError} if called when
        the producer is stopped.
        """
        self.assertRaises(ValueError, self._stopped().pauseProducing)

    def test_stopThenResume(self) -> None:
        """
        L{StrictPushProducer.resumeProducing} raises L{ValueError} if called when
        the producer is stopped.
        """
        self.assertRaises(ValueError, self._stopped().resumeProducing)

    def test_pauseThenStop(self) -> None:
        """
        L{StrictPushProducer} is stopped if C{stopProducing} is called on a paused
        producer.
        """
        producer = self._paused()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_pauseThenPause(self) -> None:
        """
        L{StrictPushProducer.pauseProducing} raises L{ValueError} if called on a
        paused producer.
        """
        producer = self._paused()
        self.assertRaises(ValueError, producer.pauseProducing)

    def test_pauseThenResume(self) -> None:
        """
        L{StrictPushProducer} is resumed if C{resumeProducing} is called on a
        paused producer.
        """
        producer = self._paused()
        producer.resumeProducing()
        self.assertRunning(producer)

    def test_resumeThenStop(self) -> None:
        """
        L{StrictPushProducer} is stopped if C{stopProducing} is called on a
        resumed producer.
        """
        producer = self._resumed()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_resumeThenPause(self) -> None:
        """
        L{StrictPushProducer} is paused if C{pauseProducing} is called on a
        resumed producer.
        """
        producer = self._resumed()
        producer.pauseProducing()
        self.assertPaused(producer)

    def test_resumeThenResume(self) -> None:
        """
        L{StrictPushProducer.resumeProducing} raises L{ValueError} if called on a
        resumed producer.
        """
        producer = self._resumed()
        self.assertRaises(ValueError, producer.resumeProducing)

    def test_stop(self) -> None:
        """
        L{StrictPushProducer} is stopped if C{stopProducing} is called in the
        initial state.
        """
        producer = self._initial()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_pause(self) -> None:
        """
        L{StrictPushProducer} is paused if C{pauseProducing} is called in the
        initial state.
        """
        producer = self._initial()
        producer.pauseProducing()
        self.assertPaused(producer)

    def test_resume(self) -> None:
        """
        L{StrictPushProducer} raises L{ValueError} if C{resumeProducing} is called
        in the initial state.
        """
        producer = self._initial()
        self.assertRaises(ValueError, producer.resumeProducing)