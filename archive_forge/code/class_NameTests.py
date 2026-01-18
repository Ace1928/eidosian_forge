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
class NameTests(unittest.TestCase):
    """
    Tests for L{Name}, the representation of a single domain name with support
    for encoding into and decoding from DNS message format.
    """

    def test_nonStringName(self):
        """
        When constructed with a name which is neither C{bytes} nor C{str},
        L{Name} raises L{TypeError}.
        """
        self.assertRaises(TypeError, dns.Name, 123)
        self.assertRaises(TypeError, dns.Name, object())
        self.assertRaises(TypeError, dns.Name, [])

    def test_unicodeName(self):
        """
        L{dns.Name} automatically encodes unicode domain name using C{idna}
        encoding.
        """
        name = dns.Name('échec.example.org')
        self.assertIsInstance(name.name, bytes)
        self.assertEqual(b'xn--chec-9oa.example.org', name.name)

    def test_decode(self):
        """
        L{Name.decode} populates the L{Name} instance with name information read
        from the file-like object passed to it.
        """
        n = dns.Name()
        n.decode(BytesIO(b'\x07example\x03com\x00'))
        self.assertEqual(n.name, b'example.com')

    def test_encode(self):
        """
        L{Name.encode} encodes its name information and writes it to the
        file-like object passed to it.
        """
        name = dns.Name(b'foo.example.com')
        stream = BytesIO()
        name.encode(stream)
        self.assertEqual(stream.getvalue(), b'\x03foo\x07example\x03com\x00')

    def test_encodeWithCompression(self):
        """
        If a compression dictionary is passed to it, L{Name.encode} uses offset
        information from it to encode its name with references to existing
        labels in the stream instead of including another copy of them in the
        output.  It also updates the compression dictionary with the location of
        the name it writes to the stream.
        """
        name = dns.Name(b'foo.example.com')
        compression = {b'example.com': 23}
        previous = b'some prefix to change .tell()'
        stream = BytesIO()
        stream.write(previous)
        expected = len(previous) + dns.Message.headerSize
        name.encode(stream, compression)
        self.assertEqual(b'\x03foo\xc0\x17', stream.getvalue()[len(previous):])
        self.assertEqual({b'example.com': 23, b'foo.example.com': expected}, compression)

    def test_unknown(self):
        """
        A resource record of unknown type and class is parsed into an
        L{UnknownRecord} instance with its data preserved, and an
        L{UnknownRecord} instance is serialized to a string equal to the one it
        was parsed from.
        """
        wire = b'\x01\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x01\x03foo\x03bar\x00\xde\xad\xbe\xef\xc0\x0c\xde\xad\xbe\xef\x00\x00\x01\x01\x00\x08somedata\x03baz\x03ban\x00\x00\x01\x00\x01\x00\x00\x01\x01\x00\x04\x01\x02\x03\x04'
        msg = dns.Message()
        msg.fromStr(wire)
        self.assertEqual(msg.queries, [dns.Query(b'foo.bar', type=57005, cls=48879)])
        self.assertEqual(msg.answers, [dns.RRHeader(b'foo.bar', type=57005, cls=48879, ttl=257, payload=dns.UnknownRecord(b'somedata', ttl=257))])
        self.assertEqual(msg.additional, [dns.RRHeader(b'baz.ban', type=dns.A, cls=dns.IN, ttl=257, payload=dns.Record_A('1.2.3.4', ttl=257))])
        enc = msg.toStr()
        self.assertEqual(enc, wire)

    def test_decodeWithCompression(self):
        """
        If the leading byte of an encoded label (in bytes read from a stream
        passed to L{Name.decode}) has its two high bits set, the next byte is
        treated as a pointer to another label in the stream and that label is
        included in the name being decoded.
        """
        stream = BytesIO(b'x' * 20 + b'\x01f\x03isi\x04arpa\x00\x03foo\xc0\x14\x03bar\xc0 ')
        stream.seek(20)
        name = dns.Name()
        name.decode(stream)
        self.assertEqual(b'f.isi.arpa', name.name)
        self.assertEqual(32, stream.tell())
        name.decode(stream)
        self.assertEqual(name.name, b'foo.f.isi.arpa')
        self.assertEqual(38, stream.tell())
        name.decode(stream)
        self.assertEqual(name.name, b'bar.foo.f.isi.arpa')
        self.assertEqual(44, stream.tell())

    def test_rejectCompressionLoop(self):
        """
        L{Name.decode} raises L{ValueError} if the stream passed to it includes
        a compression pointer which forms a loop, causing the name to be
        undecodable.
        """
        name = dns.Name()
        stream = BytesIO(b'\xc0\x00')
        self.assertRaises(ValueError, name.decode, stream)

    def test_equality(self):
        """
        L{Name} instances are equal as long as they have the same value for
        L{Name.name}, regardless of the case.
        """
        name1 = dns.Name(b'foo.bar')
        name2 = dns.Name(b'foo.bar')
        self.assertEqual(name1, name2)
        name3 = dns.Name(b'fOO.bar')
        self.assertEqual(name1, name3)

    def test_inequality(self):
        """
        L{Name} instances are not equal as long as they have different
        L{Name.name} attributes.
        """
        name1 = dns.Name(b'foo.bar')
        name2 = dns.Name(b'bar.foo')
        self.assertNotEqual(name1, name2)