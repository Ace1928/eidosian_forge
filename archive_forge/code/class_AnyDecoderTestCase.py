import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class AnyDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Any()

    def testSimple(self):
        assert decoder.decode('fox', asn1Spec=univ.Any()) == univ.Any('fox')