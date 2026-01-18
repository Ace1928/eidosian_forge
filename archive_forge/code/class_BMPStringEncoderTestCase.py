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
class BMPStringEncoderTestCase(BaseTestCase):

    def testEncoding(self):
        assert encoder.encode(char.BMPString(sys.version_info[0] == 3 and 'abc' or unicode('abc'))) == ints2octs((30, 6, 0, 97, 0, 98, 0, 99)), 'Incorrect encoding'