from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class LoopbackRelayTest(unittest.TestCase):
    """
    Test for L{twisted.protocols.loopback.LoopbackRelay}
    """

    class Receiver(Protocol):
        """
        Simple Receiver class used for testing LoopbackRelay
        """
        data = b''

        def dataReceived(self, data):
            """Accumulate received data for verification"""
            self.data += data

    def test_write(self):
        """Test to verify that the write function works as expected"""
        receiver = self.Receiver()
        relay = loopback.LoopbackRelay(receiver)
        relay.write(b'abc')
        relay.write(b'def')
        self.assertEqual(receiver.data, b'')
        relay.clearBuffer()
        self.assertEqual(receiver.data, b'abcdef')

    def test_writeSequence(self):
        """Test to verify that the writeSequence function works as expected"""
        receiver = self.Receiver()
        relay = loopback.LoopbackRelay(receiver)
        relay.writeSequence([b'The ', b'quick ', b'brown ', b'fox '])
        relay.writeSequence([b'jumps ', b'over ', b'the lazy dog'])
        self.assertEqual(receiver.data, b'')
        relay.clearBuffer()
        self.assertEqual(receiver.data, b'The quick brown fox jumps over the lazy dog')