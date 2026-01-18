from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class Receiver(Protocol):
    """
        Simple Receiver class used for testing LoopbackRelay
        """
    data = b''

    def dataReceived(self, data):
        """Accumulate received data for verification"""
        self.data += data