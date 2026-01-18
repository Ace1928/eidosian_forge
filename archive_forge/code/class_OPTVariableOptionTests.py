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
class OPTVariableOptionTests(ComparisonTestsMixin, unittest.TestCase):
    """
    Tests for L{dns._OPTVariableOption}.
    """

    def test_interface(self):
        """
        L{dns._OPTVariableOption} implements L{dns.IEncodable}.
        """
        verifyClass(dns.IEncodable, dns._OPTVariableOption)

    def test_constructorArguments(self):
        """
        L{dns._OPTVariableOption.__init__} requires code and data
        arguments which are saved as public instance attributes.
        """
        h = dns._OPTVariableOption(1, b'x')
        self.assertEqual(h.code, 1)
        self.assertEqual(h.data, b'x')

    def test_repr(self):
        """
        L{dns._OPTVariableOption.__repr__} displays the code and data
        of the option.
        """
        self.assertEqual(repr(dns._OPTVariableOption(1, b'x')), '<_OPTVariableOption code=1 data=x>')

    def test_equality(self):
        """
        Two OPTVariableOption instances compare equal if they have the same
        code and data values.
        """
        self.assertNormalEqualityImplementation(dns._OPTVariableOption(1, b'x'), dns._OPTVariableOption(1, b'x'), dns._OPTVariableOption(2, b'x'))
        self.assertNormalEqualityImplementation(dns._OPTVariableOption(1, b'x'), dns._OPTVariableOption(1, b'x'), dns._OPTVariableOption(1, b'y'))

    def test_encode(self):
        """
        L{dns._OPTVariableOption.encode} encodes the code and data
        instance attributes to a byte string which also includes the
        data length.
        """
        o = dns._OPTVariableOption(1, b'foobar')
        b = BytesIO()
        o.encode(b)
        self.assertEqual(b.getvalue(), b'\x00\x01\x00\x06foobar')

    def test_decode(self):
        """
        L{dns._OPTVariableOption.decode} is a classmethod that decodes
        a byte string and returns a L{dns._OPTVariableOption} instance.
        """
        b = BytesIO(b'\x00\x01\x00\x06foobar')
        o = dns._OPTVariableOption()
        o.decode(b)
        self.assertEqual(o.code, 1)
        self.assertEqual(o.data, b'foobar')