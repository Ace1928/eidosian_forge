from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
class FakeProducer:
    """
    A producer that does nothing.
    """

    def pauseProducing(self):
        pass

    def resumeProducing(self):
        pass

    def stopProducing(self):
        pass