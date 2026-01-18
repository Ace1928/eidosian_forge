import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class IntegerEncoderWithSchemaTestCase(BaseTestCase):

    def testPosInt(self):
        assert encoder.encode(12, asn1Spec=univ.Integer()) == ints2octs((2, 1, 12))

    def testNegInt(self):
        assert encoder.encode(-12, asn1Spec=univ.Integer()) == ints2octs((2, 1, 244))

    def testZero(self):
        assert encoder.encode(0, asn1Spec=univ.Integer()) == ints2octs((2, 1, 0))

    def testPosLong(self):
        assert encoder.encode(18446744073709551615, asn1Spec=univ.Integer()) == ints2octs((2, 9, 0, 255, 255, 255, 255, 255, 255, 255, 255))