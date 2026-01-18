import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
class ReconnectingFactoryTests(TestCase):
    """
    Tests for L{ReconnectingClientFactory}.
    """

    def test_stopTryingWhenConnected(self):
        """
        If a L{ReconnectingClientFactory} has C{stopTrying} called while it is
        connected, it does not subsequently attempt to reconnect if the
        connection is later lost.
        """

        class NoConnectConnector:

            def stopConnecting(self):
                raise RuntimeError("Shouldn't be called, we're connected.")

            def connect(self):
                raise RuntimeError("Shouldn't be reconnecting.")
        c = ReconnectingClientFactory()
        c.protocol = Protocol
        c.buildProtocol(None)
        c.stopTrying()
        c.clientConnectionLost(NoConnectConnector(), None)
        self.assertFalse(c.continueTrying)

    def test_stopTryingDoesNotReconnect(self):
        """
        Calling stopTrying on a L{ReconnectingClientFactory} doesn't attempt a
        retry on any active connector.
        """

        class FactoryAwareFakeConnector(FakeConnector):
            attemptedRetry = False

            def stopConnecting(self):
                """
                Behave as though an ongoing connection attempt has now
                failed, and notify the factory of this.
                """
                f.clientConnectionFailed(self, None)

            def connect(self):
                """
                Record an attempt to reconnect, since this is what we
                are trying to avoid.
                """
                self.attemptedRetry = True
        f = ReconnectingClientFactory()
        f.clock = Clock()
        f.connector = FactoryAwareFakeConnector()
        f.stopTrying()
        self.assertFalse(f.connector.attemptedRetry)
        self.assertFalse(f.clock.getDelayedCalls())

    def test_serializeUnused(self):
        """
        A L{ReconnectingClientFactory} which hasn't been used for anything
        can be pickled and unpickled and end up with the same state.
        """
        original = ReconnectingClientFactory()
        reconstituted = pickle.loads(pickle.dumps(original))
        self.assertEqual(original.__dict__, reconstituted.__dict__)

    def test_serializeWithClock(self):
        """
        The clock attribute of L{ReconnectingClientFactory} is not serialized,
        and the restored value sets it to the default value, the reactor.
        """
        clock = Clock()
        original = ReconnectingClientFactory()
        original.clock = clock
        reconstituted = pickle.loads(pickle.dumps(original))
        self.assertIsNone(reconstituted.clock)

    def test_deserializationResetsParameters(self):
        """
        A L{ReconnectingClientFactory} which is unpickled does not have an
        L{IConnector} and has its reconnecting timing parameters reset to their
        initial values.
        """
        factory = ReconnectingClientFactory()
        factory.clientConnectionFailed(FakeConnector(), None)
        self.addCleanup(factory.stopTrying)
        serialized = pickle.dumps(factory)
        unserialized = pickle.loads(serialized)
        self.assertIsNone(unserialized.connector)
        self.assertIsNone(unserialized._callID)
        self.assertEqual(unserialized.retries, 0)
        self.assertEqual(unserialized.delay, factory.initialDelay)
        self.assertTrue(unserialized.continueTrying)

    def test_parametrizedClock(self):
        """
        The clock used by L{ReconnectingClientFactory} can be parametrized, so
        that one can cleanly test reconnections.
        """
        clock = Clock()
        factory = ReconnectingClientFactory()
        factory.clock = clock
        factory.clientConnectionLost(FakeConnector(), None)
        self.assertEqual(len(clock.calls), 1)