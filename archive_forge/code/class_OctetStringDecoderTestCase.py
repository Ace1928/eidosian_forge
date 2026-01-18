import sys
from tests.base import BaseTestCase
from pyasn1.codec.cer import decoder
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class OctetStringDecoderTestCase(BaseTestCase):

    def testShortMode(self):
        assert decoder.decode(ints2octs((4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120))) == (str2octs('Quick brown fox'), null)

    def testLongMode(self):
        assert decoder.decode(ints2octs((36, 128, 4, 130, 3, 232) + (81,) * 1000 + (4, 1, 81, 0, 0))) == (str2octs('Q' * 1001), null)