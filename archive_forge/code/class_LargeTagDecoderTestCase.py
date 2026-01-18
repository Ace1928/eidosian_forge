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
class LargeTagDecoderTestCase(BaseTestCase):

    def testLargeTag(self):
        assert decoder.decode(ints2octs((127, 141, 245, 182, 253, 47, 3, 2, 1, 1))) == (1, null)

    def testLongTag(self):
        assert decoder.decode(ints2octs((31, 2, 1, 0)))[0].tagSet == univ.Integer.tagSet

    def testTagsEquivalence(self):
        integer = univ.Integer(2).subtype(implicitTag=tag.Tag(tag.tagClassContext, 0, 0))
        assert decoder.decode(ints2octs((159, 128, 0, 2, 1, 2)), asn1Spec=integer) == decoder.decode(ints2octs((159, 0, 2, 1, 2)), asn1Spec=integer)