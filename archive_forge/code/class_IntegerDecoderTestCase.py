import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class IntegerDecoderTestCase(BaseTestCase):

    def testPosInt(self):
        assert decoder.decode(12, asn1Spec=univ.Integer()) == univ.Integer(12)

    def testNegInt(self):
        assert decoder.decode(-12, asn1Spec=univ.Integer()) == univ.Integer(-12)