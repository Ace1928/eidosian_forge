import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class SequenceOfDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.SequenceOf(componentType=univ.OctetString())
        self.s.setComponentByPosition(0, univ.OctetString('quick brown'))

    def testDefMode(self):
        assert decoder.decode(ints2octs((48, 13, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110))) == (self.s, null)

    def testIndefMode(self):
        assert decoder.decode(ints2octs((48, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0))) == (self.s, null)

    def testDefModeChunked(self):
        assert decoder.decode(ints2octs((48, 19, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110))) == (self.s, null)

    def testIndefModeChunked(self):
        assert decoder.decode(ints2octs((48, 128, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 0, 0))) == (self.s, null)

    def testSchemalessDecoder(self):
        assert decoder.decode(ints2octs((48, 13, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110)), asn1Spec=univ.SequenceOf()) == (self.s, null)