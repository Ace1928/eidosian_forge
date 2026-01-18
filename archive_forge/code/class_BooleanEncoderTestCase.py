import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class BooleanEncoderTestCase(BaseTestCase):

    def testTrue(self):
        assert encoder.encode(univ.Boolean(1)) == ints2octs((1, 1, 255))

    def testFalse(self):
        assert encoder.encode(univ.Boolean(0)) == ints2octs((1, 1, 0))