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
class BitStringEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.b = (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1)
        self.s = univ.BitString()

    def testDefMode(self):
        assert encoder.encode(self.b, asn1Spec=self.s) == ints2octs((3, 3, 1, 169, 138))

    def testIndefMode(self):
        assert encoder.encode(self.b, asn1Spec=self.s, defMode=False) == ints2octs((3, 3, 1, 169, 138))

    def testDefModeChunked(self):
        assert encoder.encode(self.b, asn1Spec=self.s, maxChunkSize=1) == ints2octs((35, 8, 3, 2, 0, 169, 3, 2, 1, 138))

    def testIndefModeChunked(self):
        assert encoder.encode(self.b, asn1Spec=self.s, defMode=False, maxChunkSize=1) == ints2octs((35, 128, 3, 2, 0, 169, 3, 2, 1, 138, 0, 0))

    def testEmptyValue(self):
        assert encoder.encode([], asn1Spec=self.s) == ints2octs((3, 1, 0))