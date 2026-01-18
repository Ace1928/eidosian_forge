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
class ChoiceEncoderWithComponentsSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null('')), namedtype.NamedType('number', univ.Integer(0)), namedtype.NamedType('string', univ.OctetString())))

    def testEmpty(self):
        try:
            encoder.encode(self.s)
        except PyAsn1Error:
            pass
        else:
            assert 0, 'encoded unset choice'

    def testFilled(self):
        self.s.setComponentByPosition(0, univ.Null(''))
        assert encoder.encode(self.s) == ints2octs((5, 0))

    def testTagged(self):
        s = self.s.subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))
        s.setComponentByPosition(0, univ.Null(''))
        assert encoder.encode(s) == ints2octs((164, 2, 5, 0))

    def testUndefLength(self):
        self.s.setComponentByPosition(2, univ.OctetString('abcdefgh'))
        assert encoder.encode(self.s, defMode=False, maxChunkSize=3) == ints2octs((36, 128, 4, 3, 97, 98, 99, 4, 3, 100, 101, 102, 4, 2, 103, 104, 0, 0))

    def testTaggedUndefLength(self):
        s = self.s.subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))
        s.setComponentByPosition(2, univ.OctetString('abcdefgh'))
        assert encoder.encode(s, defMode=False, maxChunkSize=3) == ints2octs((164, 128, 36, 128, 4, 3, 97, 98, 99, 4, 3, 100, 101, 102, 4, 2, 103, 104, 0, 0, 0, 0))