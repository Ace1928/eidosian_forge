import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class NonStringDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null(null)), namedtype.NamedType('first-name', univ.OctetString(null)), namedtype.NamedType('age', univ.Integer(33))))
        self.s.setComponentByPosition(0, univ.Null(null))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))
        self.substrate = ints2octs([48, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1])

    def testOctetString(self):
        s, _ = decoder.decode(univ.OctetString(self.substrate), asn1Spec=self.s)
        assert self.s == s

    def testAny(self):
        s, _ = decoder.decode(univ.Any(self.substrate), asn1Spec=self.s)
        assert self.s == s