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
class OPTHeaderTests(ComparisonTestsMixin, unittest.TestCase):
    """
    Tests for L{twisted.names.dns._OPTHeader}.
    """

    def test_interface(self):
        """
        L{dns._OPTHeader} implements L{dns.IEncodable}.
        """
        verifyClass(dns.IEncodable, dns._OPTHeader)

    def test_name(self):
        """
        L{dns._OPTHeader.name} is an instance attribute whose value is
        fixed as the root domain
        """
        self.assertEqual(dns._OPTHeader().name, dns.Name(b''))

    def test_nameReadonly(self):
        """
        L{dns._OPTHeader.name} is readonly.
        """
        h = dns._OPTHeader()
        self.assertRaises(AttributeError, setattr, h, 'name', dns.Name(b'example.com'))

    def test_type(self):
        """
        L{dns._OPTHeader.type} is an instance attribute with fixed value
        41.
        """
        self.assertEqual(dns._OPTHeader().type, 41)

    def test_typeReadonly(self):
        """
        L{dns._OPTHeader.type} is readonly.
        """
        h = dns._OPTHeader()
        self.assertRaises(AttributeError, setattr, h, 'type', dns.A)

    def test_udpPayloadSize(self):
        """
        L{dns._OPTHeader.udpPayloadSize} defaults to 4096 as
        recommended in rfc6891 section-6.2.5.
        """
        self.assertEqual(dns._OPTHeader().udpPayloadSize, 4096)

    def test_udpPayloadSizeOverride(self):
        """
        L{dns._OPTHeader.udpPayloadSize} can be overridden in the
        constructor.
        """
        self.assertEqual(dns._OPTHeader(udpPayloadSize=512).udpPayloadSize, 512)

    def test_extendedRCODE(self):
        """
        L{dns._OPTHeader.extendedRCODE} defaults to 0.
        """
        self.assertEqual(dns._OPTHeader().extendedRCODE, 0)

    def test_extendedRCODEOverride(self):
        """
        L{dns._OPTHeader.extendedRCODE} can be overridden in the
        constructor.
        """
        self.assertEqual(dns._OPTHeader(extendedRCODE=1).extendedRCODE, 1)

    def test_version(self):
        """
        L{dns._OPTHeader.version} defaults to 0.
        """
        self.assertEqual(dns._OPTHeader().version, 0)

    def test_versionOverride(self):
        """
        L{dns._OPTHeader.version} can be overridden in the
        constructor.
        """
        self.assertEqual(dns._OPTHeader(version=1).version, 1)

    def test_dnssecOK(self):
        """
        L{dns._OPTHeader.dnssecOK} defaults to False.
        """
        self.assertFalse(dns._OPTHeader().dnssecOK)

    def test_dnssecOKOverride(self):
        """
        L{dns._OPTHeader.dnssecOK} can be overridden in the
        constructor.
        """
        self.assertTrue(dns._OPTHeader(dnssecOK=True).dnssecOK)

    def test_options(self):
        """
        L{dns._OPTHeader.options} defaults to empty list.
        """
        self.assertEqual(dns._OPTHeader().options, [])

    def test_optionsOverride(self):
        """
        L{dns._OPTHeader.options} can be overridden in the
        constructor.
        """
        h = dns._OPTHeader(options=[(1, 1, b'\x00')])
        self.assertEqual(h.options, [(1, 1, b'\x00')])

    def test_encode(self):
        """
        L{dns._OPTHeader.encode} packs the header fields and writes
        them to a file like object passed in as an argument.
        """
        b = BytesIO()
        OPTNonStandardAttributes.object().encode(b)
        self.assertEqual(b.getvalue(), OPTNonStandardAttributes.bytes())

    def test_encodeWithOptions(self):
        """
        L{dns._OPTHeader.options} is a list of L{dns._OPTVariableOption}
        instances which are packed into the rdata area of the header.
        """
        h = OPTNonStandardAttributes.object()
        h.options = [dns._OPTVariableOption(1, b'foobarbaz'), dns._OPTVariableOption(2, b'qux')]
        b = BytesIO()
        h.encode(b)
        self.assertEqual(b.getvalue(), OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x14\x00\x01\x00\tfoobarbaz\x00\x02\x00\x03qux')

    def test_decode(self):
        """
        L{dns._OPTHeader.decode} unpacks the header fields from a file
        like object and populates the attributes of an existing
        L{dns._OPTHeader} instance.
        """
        decodedHeader = dns._OPTHeader()
        decodedHeader.decode(BytesIO(OPTNonStandardAttributes.bytes()))
        self.assertEqual(decodedHeader, OPTNonStandardAttributes.object())

    def test_decodeAllExpectedBytes(self):
        """
        L{dns._OPTHeader.decode} reads all the bytes of the record
        that is being decoded.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes())
        decodedHeader = dns._OPTHeader()
        decodedHeader.decode(b)
        self.assertEqual(b.tell(), len(b.getvalue()))

    def test_decodeOnlyExpectedBytes(self):
        """
        L{dns._OPTHeader.decode} reads only the bytes from the current
        file position to the end of the record that is being
        decoded. Trailing bytes are not consumed.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes() + b'xxxx')
        decodedHeader = dns._OPTHeader()
        decodedHeader.decode(b)
        self.assertEqual(b.tell(), len(b.getvalue()) - len(b'xxxx'))

    def test_decodeDiscardsName(self):
        """
        L{dns._OPTHeader.decode} discards the name which is encoded in
        the supplied bytes. The name attribute of the resulting
        L{dns._OPTHeader} instance will always be L{dns.Name(b'')}.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes(excludeName=True) + b'\x07example\x03com\x00')
        h = dns._OPTHeader()
        h.decode(b)
        self.assertEqual(h.name, dns.Name(b''))

    def test_decodeRdlengthTooShort(self):
        """
        L{dns._OPTHeader.decode} raises an exception if the supplied
        RDLEN is too short.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x05\x00\x01\x00\x02\x00\x00')
        h = dns._OPTHeader()
        self.assertRaises(EOFError, h.decode, b)

    def test_decodeRdlengthTooLong(self):
        """
        L{dns._OPTHeader.decode} raises an exception if the supplied
        RDLEN is too long.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x07\x00\x01\x00\x02\x00\x00')
        h = dns._OPTHeader()
        self.assertRaises(EOFError, h.decode, b)

    def test_decodeWithOptions(self):
        """
        If the OPT bytes contain variable options,
        L{dns._OPTHeader.decode} will populate a list
        L{dns._OPTHeader.options} with L{dns._OPTVariableOption}
        instances.
        """
        b = BytesIO(OPTNonStandardAttributes.bytes(excludeOptions=True) + b'\x00\x14\x00\x01\x00\tfoobarbaz\x00\x02\x00\x03qux')
        h = dns._OPTHeader()
        h.decode(b)
        self.assertEqual(h.options, [dns._OPTVariableOption(1, b'foobarbaz'), dns._OPTVariableOption(2, b'qux')])

    def test_fromRRHeader(self):
        """
        L{_OPTHeader.fromRRHeader} accepts an L{RRHeader} instance and
        returns an L{_OPTHeader} instance whose attribute values have
        been derived from the C{cls}, C{ttl} and C{payload} attributes
        of the original header.
        """
        genericHeader = dns.RRHeader(b'example.com', type=dns.OPT, cls=65535, ttl=254 << 24 | 253 << 16 | True << 15, payload=dns.UnknownRecord(b'\xff\xff\x00\x03abc'))
        decodedOptHeader = dns._OPTHeader.fromRRHeader(genericHeader)
        expectedOptHeader = dns._OPTHeader(udpPayloadSize=65535, extendedRCODE=254, version=253, dnssecOK=True, options=[dns._OPTVariableOption(code=65535, data=b'abc')])
        self.assertEqual(decodedOptHeader, expectedOptHeader)

    def test_repr(self):
        """
        L{dns._OPTHeader.__repr__} displays the name and type and all
        the fixed and extended header values of the OPT record.
        """
        self.assertEqual(repr(dns._OPTHeader()), '<_OPTHeader name= type=41 udpPayloadSize=4096 extendedRCODE=0 version=0 dnssecOK=False options=[]>')

    def test_equalityUdpPayloadSize(self):
        """
        Two L{OPTHeader} instances compare equal if they have the same
        udpPayloadSize.
        """
        self.assertNormalEqualityImplementation(dns._OPTHeader(udpPayloadSize=512), dns._OPTHeader(udpPayloadSize=512), dns._OPTHeader(udpPayloadSize=4096))

    def test_equalityExtendedRCODE(self):
        """
        Two L{OPTHeader} instances compare equal if they have the same
        extendedRCODE.
        """
        self.assertNormalEqualityImplementation(dns._OPTHeader(extendedRCODE=1), dns._OPTHeader(extendedRCODE=1), dns._OPTHeader(extendedRCODE=2))

    def test_equalityVersion(self):
        """
        Two L{OPTHeader} instances compare equal if they have the same
        version.
        """
        self.assertNormalEqualityImplementation(dns._OPTHeader(version=1), dns._OPTHeader(version=1), dns._OPTHeader(version=2))

    def test_equalityDnssecOK(self):
        """
        Two L{OPTHeader} instances compare equal if they have the same
        dnssecOK flags.
        """
        self.assertNormalEqualityImplementation(dns._OPTHeader(dnssecOK=True), dns._OPTHeader(dnssecOK=True), dns._OPTHeader(dnssecOK=False))

    def test_equalityOptions(self):
        """
        Two L{OPTHeader} instances compare equal if they have the same
        options.
        """
        self.assertNormalEqualityImplementation(dns._OPTHeader(options=[dns._OPTVariableOption(1, b'x')]), dns._OPTHeader(options=[dns._OPTVariableOption(1, b'x')]), dns._OPTHeader(options=[dns._OPTVariableOption(2, b'y')]))