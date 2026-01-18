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
class NetstringReceiverTests(unittest.SynchronousTestCase, LPTestCaseMixin):
    """
    Tests for L{twisted.protocols.basic.NetstringReceiver}.
    """
    strings = [b'hello', b'world', b'how', b'are', b'you123', b':today', b'a' * 515]
    illegalStrings = [b'9999999999999999999999', b'abc', b'4:abcde', b'51:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab,']
    protocol = TestNetstring

    def setUp(self):
        self.transport = proto_helpers.StringTransport()
        self.netstringReceiver = TestNetstring()
        self.netstringReceiver.makeConnection(self.transport)

    def test_buffer(self):
        """
        Strings can be received in chunks of different lengths.
        """
        for packet_size in range(1, 10):
            t = proto_helpers.StringTransport()
            a = TestNetstring()
            a.MAX_LENGTH = 699
            a.makeConnection(t)
            for s in self.strings:
                a.sendString(s)
            out = t.value()
            for i in range(len(out) // packet_size + 1):
                s = out[i * packet_size:(i + 1) * packet_size]
                if s:
                    a.dataReceived(s)
            self.assertEqual(a.received, self.strings)

    def test_receiveEmptyNetstring(self):
        """
        Empty netstrings (with length '0') can be received.
        """
        self.netstringReceiver.dataReceived(b'0:,')
        self.assertEqual(self.netstringReceiver.received, [b''])

    def test_receiveOneCharacter(self):
        """
        One-character netstrings can be received.
        """
        self.netstringReceiver.dataReceived(b'1:a,')
        self.assertEqual(self.netstringReceiver.received, [b'a'])

    def test_receiveTwoCharacters(self):
        """
        Two-character netstrings can be received.
        """
        self.netstringReceiver.dataReceived(b'2:ab,')
        self.assertEqual(self.netstringReceiver.received, [b'ab'])

    def test_receiveNestedNetstring(self):
        """
        Netstrings with embedded netstrings. This test makes sure that
        the parser does not become confused about the ',' and ':'
        characters appearing inside the data portion of the netstring.
        """
        self.netstringReceiver.dataReceived(b'4:1:a,,')
        self.assertEqual(self.netstringReceiver.received, [b'1:a,'])

    def test_moreDataThanSpecified(self):
        """
        Netstrings containing more data than expected are refused.
        """
        self.netstringReceiver.dataReceived(b'2:aaa,')
        self.assertTrue(self.transport.disconnecting)

    def test_moreDataThanSpecifiedBorderCase(self):
        """
        Netstrings that should be empty according to their length
        specification are refused if they contain data.
        """
        self.netstringReceiver.dataReceived(b'0:a,')
        self.assertTrue(self.transport.disconnecting)

    def test_missingNumber(self):
        """
        Netstrings without leading digits that specify the length
        are refused.
        """
        self.netstringReceiver.dataReceived(b':aaa,')
        self.assertTrue(self.transport.disconnecting)

    def test_missingColon(self):
        """
        Netstrings without a colon between length specification and
        data are refused.
        """
        self.netstringReceiver.dataReceived(b'3aaa,')
        self.assertTrue(self.transport.disconnecting)

    def test_missingNumberAndColon(self):
        """
        Netstrings that have no leading digits nor a colon are
        refused.
        """
        self.netstringReceiver.dataReceived(b'aaa,')
        self.assertTrue(self.transport.disconnecting)

    def test_onlyData(self):
        """
        Netstrings consisting only of data are refused.
        """
        self.netstringReceiver.dataReceived(b'aaa')
        self.assertTrue(self.transport.disconnecting)

    def test_receiveNetstringPortions_1(self):
        """
        Netstrings can be received in two portions.
        """
        self.netstringReceiver.dataReceived(b'4:aa')
        self.netstringReceiver.dataReceived(b'aa,')
        self.assertEqual(self.netstringReceiver.received, [b'aaaa'])
        self.assertTrue(self.netstringReceiver._payloadComplete())

    def test_receiveNetstringPortions_2(self):
        """
        Netstrings can be received in more than two portions, even if
        the length specification is split across two portions.
        """
        for part in [b'1', b'0:01234', b'56789', b',']:
            self.netstringReceiver.dataReceived(part)
        self.assertEqual(self.netstringReceiver.received, [b'0123456789'])

    def test_receiveNetstringPortions_3(self):
        """
        Netstrings can be received one character at a time.
        """
        for part in [b'2', b':', b'a', b'b', b',']:
            self.netstringReceiver.dataReceived(part)
        self.assertEqual(self.netstringReceiver.received, [b'ab'])

    def test_receiveTwoNetstrings(self):
        """
        A stream of two netstrings can be received in two portions,
        where the first portion contains the complete first netstring
        and the length specification of the second netstring.
        """
        self.netstringReceiver.dataReceived(b'1:a,1')
        self.assertTrue(self.netstringReceiver._payloadComplete())
        self.assertEqual(self.netstringReceiver.received, [b'a'])
        self.netstringReceiver.dataReceived(b':b,')
        self.assertEqual(self.netstringReceiver.received, [b'a', b'b'])

    def test_maxReceiveLimit(self):
        """
        Netstrings with a length specification exceeding the specified
        C{MAX_LENGTH} are refused.
        """
        tooLong = self.netstringReceiver.MAX_LENGTH + 1
        self.netstringReceiver.dataReceived(b''.join((bytes(tooLong), b':', b'a' * tooLong)))
        self.assertTrue(self.transport.disconnecting)

    def test_consumeLength(self):
        """
        C{_consumeLength} returns the expected length of the
        netstring, including the trailing comma.
        """
        self.netstringReceiver._remainingData = b'12:'
        self.netstringReceiver._consumeLength()
        self.assertEqual(self.netstringReceiver._expectedPayloadSize, 13)

    def test_consumeLengthBorderCase1(self):
        """
        C{_consumeLength} works as expected if the length specification
        contains the value of C{MAX_LENGTH} (border case).
        """
        self.netstringReceiver._remainingData = b'12:'
        self.netstringReceiver.MAX_LENGTH = 12
        self.netstringReceiver._consumeLength()
        self.assertEqual(self.netstringReceiver._expectedPayloadSize, 13)

    def test_consumeLengthBorderCase2(self):
        """
        C{_consumeLength} raises a L{basic.NetstringParseError} if
        the length specification exceeds the value of C{MAX_LENGTH}
        by 1 (border case).
        """
        self.netstringReceiver._remainingData = b'12:'
        self.netstringReceiver.MAX_LENGTH = 11
        self.assertRaises(basic.NetstringParseError, self.netstringReceiver._consumeLength)

    def test_consumeLengthBorderCase3(self):
        """
        C{_consumeLength} raises a L{basic.NetstringParseError} if
        the length specification exceeds the value of C{MAX_LENGTH}
        by more than 1.
        """
        self.netstringReceiver._remainingData = b'1000:'
        self.netstringReceiver.MAX_LENGTH = 11
        self.assertRaises(basic.NetstringParseError, self.netstringReceiver._consumeLength)

    def test_stringReceivedNotImplemented(self):
        """
        When L{NetstringReceiver.stringReceived} is not overridden in a
        subclass, calling it raises C{NotImplementedError}.
        """
        proto = basic.NetstringReceiver()
        self.assertRaises(NotImplementedError, proto.stringReceived, 'foo')