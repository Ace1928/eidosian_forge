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
class ExpTaggedSequenceEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('number', univ.Integer())))
        s = s.subtype(explicitTag=tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 5))
        s[0] = 12
        self.s = s

    def testDefMode(self):
        assert encoder.encode(self.s) == ints2octs((101, 5, 48, 3, 2, 1, 12))

    def testIndefMode(self):
        assert encoder.encode(self.s, defMode=False) == ints2octs((101, 128, 48, 128, 2, 1, 12, 0, 0, 0, 0))