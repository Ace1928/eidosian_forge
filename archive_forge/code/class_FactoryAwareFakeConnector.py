import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
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