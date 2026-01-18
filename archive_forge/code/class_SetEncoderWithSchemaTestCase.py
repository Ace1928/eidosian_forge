import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class SetEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Set(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null('')), namedtype.OptionalNamedType('first-name', univ.OctetString()), namedtype.DefaultedNamedType('age', univ.Integer(33))))

    def __init(self):
        self.s.clear()
        self.s.setComponentByPosition(0)

    def __initWithOptional(self):
        self.s.clear()
        self.s.setComponentByPosition(0)
        self.s.setComponentByPosition(1, 'quick brown')

    def __initWithDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0)
        self.s.setComponentByPosition(2, 1)

    def __initWithOptionalAndDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(''))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def testIndefMode(self):
        self.__init()
        assert encoder.encode(self.s) == ints2octs((49, 128, 5, 0, 0, 0))

    def testWithOptionalIndefMode(self):
        self.__initWithOptional()
        assert encoder.encode(self.s) == ints2octs((49, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))

    def testWithDefaultedIndefMode(self):
        self.__initWithDefaulted()
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 5, 0, 0, 0))

    def testWithOptionalAndDefaultedIndefMode(self):
        self.__initWithOptionalAndDefaulted()
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))