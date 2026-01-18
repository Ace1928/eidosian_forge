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
class OctetStringEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.OctetString()
        self.o = 'Quick brown fox'

    def testDefMode(self):
        assert encoder.encode(self.o, asn1Spec=self.s) == ints2octs((4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120))

    def testIndefMode(self):
        assert encoder.encode(self.o, asn1Spec=self.s, defMode=False) == ints2octs((4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120))

    def testDefModeChunked(self):
        assert encoder.encode(self.o, asn1Spec=self.s, maxChunkSize=4) == ints2octs((36, 23, 4, 4, 81, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 4, 111, 119, 110, 32, 4, 3, 102, 111, 120))

    def testIndefModeChunked(self):
        assert encoder.encode(self.o, asn1Spec=self.s, defMode=False, maxChunkSize=4) == ints2octs((36, 128, 4, 4, 81, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 4, 111, 119, 110, 32, 4, 3, 102, 111, 120, 0, 0))