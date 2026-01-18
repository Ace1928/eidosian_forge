import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class Int16Tests(unittest.SynchronousTestCase, IntNTestCaseMixin, RecvdAttributeMixin):
    """
    Test case for int16-prefixed protocol
    """
    protocol = TestInt16
    strings = [b'a', b'b' * 16]
    illegalStrings = [b'\x10\x00aaaaaa']
    partialStrings = [b'\x00', b'hello there', b'']

    def test_data(self):
        """
        Test specific behavior of the 16-bits length.
        """
        r = self.getProtocol()
        r.sendString(b'foo')
        self.assertEqual(r.transport.value(), b'\x00\x03foo')
        r.dataReceived(b'\x00\x04ubar')
        self.assertEqual(r.received, [b'ubar'])

    def test_tooLongSend(self):
        """
        Send too much data: that should cause an error.
        """
        r = self.getProtocol()
        tooSend = b'b' * (2 ** (r.prefixLength * 8) + 1)
        self.assertRaises(AssertionError, r.sendString, tooSend)