import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class MPTests(unittest.TestCase):
    """
    Tests for L{common.getMP}.

    @cvar getMP: a method providing a MP parser.
    @type getMP: C{callable}
    """
    if not cryptography:
        skip = "can't run without cryptography"
    if cryptography:
        getMP = staticmethod(common.getMP)

    def test_getMP(self):
        """
        L{common.getMP} should parse the a multiple precision integer from a
        string: a 4-byte length followed by length bytes of the integer.
        """
        self.assertEqual(self.getMP(b'\x00\x00\x00\x04\x00\x00\x00\x01'), (1, b''))

    def test_getMPBigInteger(self):
        """
        L{common.getMP} should be able to parse a big enough integer
        (that doesn't fit on one byte).
        """
        self.assertEqual(self.getMP(b'\x00\x00\x00\x04\x01\x02\x03\x04'), (16909060, b''))

    def test_multipleGetMP(self):
        """
        L{common.getMP} has the ability to parse multiple integer in the same
        string.
        """
        self.assertEqual(self.getMP(b'\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x02', 2), (1, 2, b''))

    def test_getMPRemainingData(self):
        """
        When more data than needed is sent to L{common.getMP}, it should return
        the remaining data.
        """
        self.assertEqual(self.getMP(b'\x00\x00\x00\x04\x00\x00\x00\x01foo'), (1, b'foo'))

    def test_notEnoughData(self):
        """
        When the string passed to L{common.getMP} doesn't even make 5 bytes,
        it should raise a L{struct.error}.
        """
        self.assertRaises(struct.error, self.getMP, b'\x02\x00')