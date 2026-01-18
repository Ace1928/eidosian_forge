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
class SequenceDecoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null(null)), namedtype.OptionalNamedType('first-name', univ.OctetString()), namedtype.DefaultedNamedType('age', univ.Integer(33))))

    def __init(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(null))

    def __initWithOptional(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(null))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))

    def __initWithDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(null))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def __initWithOptionalAndDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(null))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def testDefMode(self):
        self.__init()
        assert decoder.decode(ints2octs((48, 2, 5, 0)), asn1Spec=self.s) == (self.s, null)

    def testIndefMode(self):
        self.__init()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testDefModeChunked(self):
        self.__init()
        assert decoder.decode(ints2octs((48, 2, 5, 0)), asn1Spec=self.s) == (self.s, null)

    def testIndefModeChunked(self):
        self.__init()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalDefMode(self):
        self.__initWithOptional()
        assert decoder.decode(ints2octs((48, 15, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionaIndefMode(self):
        self.__initWithOptional()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 36, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalDefModeChunked(self):
        self.__initWithOptional()
        assert decoder.decode(ints2octs((48, 21, 5, 0, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalIndefModeChunked(self):
        self.__initWithOptional()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithDefaultedDefMode(self):
        self.__initWithDefaulted()
        assert decoder.decode(ints2octs((48, 5, 5, 0, 2, 1, 1)), asn1Spec=self.s) == (self.s, null)

    def testWithDefaultedIndefMode(self):
        self.__initWithDefaulted()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 2, 1, 1, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithDefaultedDefModeChunked(self):
        self.__initWithDefaulted()
        assert decoder.decode(ints2octs((48, 5, 5, 0, 2, 1, 1)), asn1Spec=self.s) == (self.s, null)

    def testWithDefaultedIndefModeChunked(self):
        self.__initWithDefaulted()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 2, 1, 1, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalAndDefaultedDefMode(self):
        self.__initWithOptionalAndDefaulted()
        assert decoder.decode(ints2octs((48, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalAndDefaultedIndefMode(self):
        self.__initWithOptionalAndDefaulted()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 36, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalAndDefaultedDefModeChunked(self):
        self.__initWithOptionalAndDefaulted()
        assert decoder.decode(ints2octs((48, 24, 5, 0, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 2, 1, 1)), asn1Spec=self.s) == (self.s, null)

    def testWithOptionalAndDefaultedIndefModeChunked(self):
        self.__initWithOptionalAndDefaulted()
        assert decoder.decode(ints2octs((48, 128, 5, 0, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0)), asn1Spec=self.s) == (self.s, null)