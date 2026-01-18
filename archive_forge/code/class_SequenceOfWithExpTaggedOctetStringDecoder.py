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
class SequenceOfWithExpTaggedOctetStringDecoder(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.SequenceOf(componentType=univ.OctetString().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
        self.s.setComponentByPosition(0, 'q')
        self.s2 = univ.SequenceOf()

    def testDefModeSchema(self):
        s, r = decoder.decode(ints2octs((48, 5, 163, 3, 4, 1, 113)), asn1Spec=self.s)
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet

    def testIndefModeSchema(self):
        s, r = decoder.decode(ints2octs((48, 128, 163, 128, 4, 1, 113, 0, 0, 0, 0)), asn1Spec=self.s)
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet

    def testDefModeNoComponent(self):
        s, r = decoder.decode(ints2octs((48, 5, 163, 3, 4, 1, 113)), asn1Spec=self.s2)
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet

    def testIndefModeNoComponent(self):
        s, r = decoder.decode(ints2octs((48, 128, 163, 128, 4, 1, 113, 0, 0, 0, 0)), asn1Spec=self.s2)
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet

    def testDefModeSchemaless(self):
        s, r = decoder.decode(ints2octs((48, 5, 163, 3, 4, 1, 113)))
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet

    def testIndefModeSchemaless(self):
        s, r = decoder.decode(ints2octs((48, 128, 163, 128, 4, 1, 113, 0, 0, 0, 0)))
        assert not r
        assert s == self.s
        assert s.tagSet == self.s.tagSet