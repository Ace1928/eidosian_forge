import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
class BitStringTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.b = univ.BitString(namedValues=namedval.NamedValues(('Active', 0), ('Urgent', 1)))

    def testBinDefault(self):

        class BinDefault(univ.BitString):
            defaultBinValue = '1010100110001010'
        assert BinDefault() == univ.BitString(binValue='1010100110001010')

    def testHexDefault(self):

        class HexDefault(univ.BitString):
            defaultHexValue = 'A98A'
        assert HexDefault() == univ.BitString(hexValue='A98A')

    def testSet(self):
        assert self.b.clone('Active') == (1,)
        assert self.b.clone('Urgent') == (0, 1)
        assert self.b.clone('Urgent, Active') == (1, 1)
        assert self.b.clone("'1010100110001010'B") == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
        assert self.b.clone("'A98A'H") == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
        assert self.b.clone(binValue='1010100110001010') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
        assert self.b.clone(hexValue='A98A') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
        assert self.b.clone('1010100110001010') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
        assert self.b.clone((1, 0, 1)) == (1, 0, 1)

    def testStr(self):
        assert str(self.b.clone('Urgent')) == '01'

    def testRepr(self):
        assert 'BitString' in repr(self.b.clone('Urgent,Active'))

    def testTag(self):
        assert univ.BitString().tagSet == tag.TagSet((), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 3))

    def testLen(self):
        assert len(self.b.clone("'A98A'H")) == 16

    def testGetItem(self):
        assert self.b.clone("'A98A'H")[0] == 1
        assert self.b.clone("'A98A'H")[1] == 0
        assert self.b.clone("'A98A'H")[2] == 1
    if sys.version_info[:2] > (2, 4):

        def testReverse(self):
            assert list(reversed(univ.BitString([0, 0, 1]))) == list(univ.BitString([1, 0, 0]))

    def testAsOctets(self):
        assert self.b.clone(hexValue='A98A').asOctets() == ints2octs((169, 138)), 'testAsOctets() fails'

    def testAsInts(self):
        assert self.b.clone(hexValue='A98A').asNumbers() == (169, 138), 'testAsNumbers() fails'

    def testMultipleOfEightPadding(self):
        assert self.b.clone((1, 0, 1)).asNumbers() == (5,)

    def testAsInteger(self):
        assert self.b.clone('11000000011001').asInteger() == 12313
        assert self.b.clone('1100110011011111').asInteger() == 52447

    def testStaticDef(self):

        class BitString(univ.BitString):
            pass
        assert BitString('11000000011001').asInteger() == 12313