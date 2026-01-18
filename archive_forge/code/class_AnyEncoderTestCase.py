import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import encoder
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class AnyEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Any(encoder.encode(univ.OctetString('fox')))

    def testSimple(self):
        assert encoder.encode(self.s) == str2octs('fox')