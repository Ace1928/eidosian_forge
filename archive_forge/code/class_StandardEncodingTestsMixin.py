import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class StandardEncodingTestsMixin:
    """
    Tests for the encoding and decoding of various standard (not EDNS) messages.

    These tests should work with both L{dns._EDNSMessage} and L{dns.Message}.

    TestCase classes that use this mixin must provide a C{messageFactory} method
    which accepts any argment supported by L{dns._EDNSMessage.__init__}.

    EDNS specific arguments may be discarded if not supported by the message
    class under construction.
    """

    def test_emptyMessageEncode(self):
        """
        An empty message can be encoded.
        """
        self.assertEqual(self.messageFactory(**MessageEmpty.kwargs()).toStr(), MessageEmpty.bytes())

    def test_emptyMessageDecode(self):
        """
        An empty message byte sequence can be decoded.
        """
        m = self.messageFactory()
        m.fromStr(MessageEmpty.bytes())
        self.assertEqual(m, self.messageFactory(**MessageEmpty.kwargs()))

    def test_completeQueryEncode(self):
        """
        A fully populated query message can be encoded.
        """
        self.assertEqual(self.messageFactory(**MessageComplete.kwargs()).toStr(), MessageComplete.bytes())

    def test_completeQueryDecode(self):
        """
        A fully populated message byte string can be decoded.
        """
        m = self.messageFactory()
        (m.fromStr(MessageComplete.bytes()),)
        self.assertEqual(m, self.messageFactory(**MessageComplete.kwargs()))

    def test_NULL(self):
        """
        A I{NULL} record with an arbitrary payload can be encoded and decoded as
        part of a message.
        """
        bytes = b''.join([dns._ord2bytes(i) for i in range(256)])
        rec = dns.Record_NULL(bytes)
        rr = dns.RRHeader(b'testname', dns.NULL, payload=rec)
        msg1 = self.messageFactory()
        msg1.answers.append(rr)
        s = msg1.toStr()
        msg2 = self.messageFactory()
        msg2.fromStr(s)
        self.assertIsInstance(msg2.answers[0].payload, dns.Record_NULL)
        self.assertEqual(msg2.answers[0].payload.payload, bytes)

    def test_nonAuthoritativeMessageEncode(self):
        """
        If the message C{authoritative} attribute is set to 0, the encoded bytes
        will have AA bit 0.
        """
        self.assertEqual(self.messageFactory(**MessageNonAuthoritative.kwargs()).toStr(), MessageNonAuthoritative.bytes())

    def test_nonAuthoritativeMessageDecode(self):
        """
        The L{dns.RRHeader} instances created by a message from a
        non-authoritative message byte string are marked as not authoritative.
        """
        m = self.messageFactory()
        m.fromStr(MessageNonAuthoritative.bytes())
        self.assertEqual(m, self.messageFactory(**MessageNonAuthoritative.kwargs()))

    def test_authoritativeMessageEncode(self):
        """
        If the message C{authoritative} attribute is set to 1, the encoded bytes
        will have AA bit 1.
        """
        self.assertEqual(self.messageFactory(**MessageAuthoritative.kwargs()).toStr(), MessageAuthoritative.bytes())

    def test_authoritativeMessageDecode(self):
        """
        The message and its L{dns.RRHeader} instances created by C{decode} from
        an authoritative message byte string, are marked as authoritative.
        """
        m = self.messageFactory()
        m.fromStr(MessageAuthoritative.bytes())
        self.assertEqual(m, self.messageFactory(**MessageAuthoritative.kwargs()))

    def test_truncatedMessageEncode(self):
        """
        If the message C{trunc} attribute is set to 1 the encoded bytes will
        have TR bit 1.
        """
        self.assertEqual(self.messageFactory(**MessageTruncated.kwargs()).toStr(), MessageTruncated.bytes())

    def test_truncatedMessageDecode(self):
        """
        The message instance created by decoding a truncated message is marked
        as truncated.
        """
        m = self.messageFactory()
        m.fromStr(MessageTruncated.bytes())
        self.assertEqual(m, self.messageFactory(**MessageTruncated.kwargs()))