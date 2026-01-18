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
class SequenceEncoderWithImplicitlyTaggedOpenTypesTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        openType = opentype.OpenType('id', {1: univ.Integer(), 2: univ.OctetString()})
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('id', univ.Integer()), namedtype.NamedType('blob', univ.Any().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)), openType=openType)))

    def testEncodeOpenTypeChoiceOne(self):
        self.s.clear()
        self.s[0] = 1
        self.s[1] = univ.Integer(12)
        assert encoder.encode(self.s, asn1Spec=self.s) == ints2octs((48, 8, 2, 1, 1, 131, 3, 2, 1, 12))