import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class NestedOptionalChoiceEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        layer3 = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('first-name', univ.OctetString()), namedtype.DefaultedNamedType('age', univ.Integer(33))))
        layer2 = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('inner', layer3), namedtype.NamedType('first-name', univ.OctetString())))
        layer1 = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('inner', layer2)))
        self.s = layer1

    def __initOptionalWithDefaultAndOptional(self):
        self.s.clear()
        self.s[0][0][0] = 'test'
        self.s[0][0][1] = 123
        return self.s

    def __initOptionalWithDefault(self):
        self.s.clear()
        self.s[0][0][1] = 123
        return self.s

    def __initOptionalWithOptional(self):
        self.s.clear()
        self.s[0][0][0] = 'test'
        return self.s

    def __initOptional(self):
        self.s.clear()
        return self.s

    def testOptionalWithDefaultAndOptional(self):
        s = self.__initOptionalWithDefaultAndOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 2, 1, 123, 0, 0, 0, 0))

    def testOptionalWithDefault(self):
        s = self.__initOptionalWithDefault()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 2, 1, 123, 0, 0, 0, 0))

    def testOptionalWithOptional(self):
        s = self.__initOptionalWithOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 0, 0, 0, 0))

    def testOptional(self):
        s = self.__initOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 0, 0))