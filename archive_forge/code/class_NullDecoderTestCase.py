import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class NullDecoderTestCase(BaseTestCase):

    def testNull(self):
        assert decoder.decode(None, asn1Spec=univ.Null()) == univ.Null('')