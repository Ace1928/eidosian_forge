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
class ChoiceEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null('')), namedtype.NamedType('number', univ.Integer(0)), namedtype.NamedType('string', univ.OctetString())))
        self.v = {'place-holder': None}

    def testFilled(self):
        assert encoder.encode(self.v, asn1Spec=self.s) == ints2octs((5, 0))