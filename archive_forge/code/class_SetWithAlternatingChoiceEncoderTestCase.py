import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
class SetWithAlternatingChoiceEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        c = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('name', univ.OctetString()), namedtype.NamedType('amount', univ.Boolean())))
        self.s = univ.Set(componentType=namedtype.NamedTypes(namedtype.NamedType('value', univ.Integer(5)), namedtype.NamedType('status', c)))

    def testComponentsOrdering1(self):
        self.s.setComponentByName('status')
        self.s.getComponentByName('status').setComponentByPosition(0, 'A')
        assert encoder.encode(self.s) == ints2octs((49, 6, 2, 1, 5, 4, 1, 65))

    def testComponentsOrdering2(self):
        self.s.setComponentByName('status')
        self.s.getComponentByName('status').setComponentByPosition(1, True)
        assert encoder.encode(self.s) == ints2octs((49, 6, 1, 1, 255, 2, 1, 5))