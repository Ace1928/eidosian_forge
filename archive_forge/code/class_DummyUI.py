from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
class DummyUI:
    """
    Provide just the interface required to be passed to AbstractAccount.logOn.
    """
    clientRegistered = False

    def registerAccountClient(self, result):
        self.clientRegistered = True