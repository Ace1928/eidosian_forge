import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class SequenceDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null()), namedtype.NamedType('first-name', univ.OctetString()), namedtype.NamedType('age', univ.Integer(33))))

    def testSimple(self):
        s = self.s.clone()
        s[0] = univ.Null('')
        s[1] = univ.OctetString('xx')
        s[2] = univ.Integer(33)
        assert decoder.decode({'place-holder': None, 'first-name': 'xx', 'age': 33}, asn1Spec=self.s) == s