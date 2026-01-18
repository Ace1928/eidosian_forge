import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import encoder
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class BadAsn1SpecTestCase(BaseTestCase):

    def testBadValueType(self):
        try:
            encoder.encode('not an Asn1Item')
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Invalid value type accepted'