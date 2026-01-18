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
class ExpTaggedSequenceComponentEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('number', univ.Boolean().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))))
        self.s[0] = True

    def testDefMode(self):
        assert encoder.encode(self.s) == ints2octs((48, 5, 160, 3, 1, 1, 1))

    def testIndefMode(self):
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 160, 3, 1, 1, 1, 0, 0, 0, 0))