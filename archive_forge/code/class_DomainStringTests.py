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
class DomainStringTests(unittest.SynchronousTestCase):

    def test_bytes(self):
        """
        L{dns.domainString} returns L{bytes} unchanged.
        """
        self.assertEqual(b'twistedmatrix.com', dns.domainString(b'twistedmatrix.com'))

    def test_native(self):
        """
        L{dns.domainString} converts a native string to L{bytes}
        if necessary.
        """
        self.assertEqual(b'example.com', dns.domainString('example.com'))

    def test_text(self):
        """
        L{dns.domainString} always converts a unicode string to L{bytes}.
        """
        self.assertEqual(b'foo.example', dns.domainString('foo.example'))

    def test_idna(self):
        """
        L{dns.domainString} encodes Unicode using IDNA.
        """
        self.assertEqual(b'xn--fwg.test', dns.domainString('‽.test'))

    def test_nonsense(self):
        """
        L{dns.domainString} encodes Unicode using IDNA.
        """
        self.assertRaises(TypeError, dns.domainString, 9000)
        self.assertRaises(TypeError, dns.domainString, dns.Name('bar.example'))