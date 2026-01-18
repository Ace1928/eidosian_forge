import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
class FakeConnector:
    """
    A fake connector class, to be used to mock connections failed or lost.
    """

    def stopConnecting(self):
        pass

    def connect(self):
        pass