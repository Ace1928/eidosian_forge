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
class ObjectIdentifierWithSchemaEncoderTestCase(BaseTestCase):

    def testOne(self):
        assert encoder.encode((1, 3, 6, 0, 1048574), asn1Spec=univ.ObjectIdentifier()) == ints2octs((6, 6, 43, 6, 0, 191, 255, 126))