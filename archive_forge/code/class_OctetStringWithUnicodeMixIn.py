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
class OctetStringWithUnicodeMixIn(object):
    initializer = ()
    encoding = 'us-ascii'

    def setUp(self):
        self.pythonString = ints2octs(self.initializer).decode(self.encoding)
        self.encodedPythonString = self.pythonString.encode(self.encoding)
        self.numbersString = tuple(octs2ints(self.encodedPythonString))

    def testInit(self):
        assert univ.OctetString(self.encodedPythonString) == self.encodedPythonString, '__init__() fails'

    def testInitFromAsn1(self):
        assert univ.OctetString(univ.OctetString(self.encodedPythonString)) == self.encodedPythonString
        assert univ.OctetString(univ.Integer(123)) == univ.OctetString('123')

    def testSerialised(self):
        if sys.version_info[0] < 3:
            assert str(univ.OctetString(self.encodedPythonString, encoding=self.encoding)) == self.encodedPythonString, '__str__() fails'
        else:
            assert bytes(univ.OctetString(self.encodedPythonString, encoding=self.encoding)) == self.encodedPythonString, '__str__() fails'

    def testPrintable(self):
        if sys.version_info[0] < 3:
            assert str(univ.OctetString(self.encodedPythonString, encoding=self.encoding)) == self.encodedPythonString, '__str__() fails'
            assert unicode(univ.OctetString(self.pythonString, encoding=self.encoding)) == self.pythonString, 'unicode init fails'
        else:
            assert str(univ.OctetString(self.pythonString, encoding=self.encoding)) == self.pythonString, 'unicode init fails'

    def testSeq(self):
        assert univ.OctetString(self.encodedPythonString)[0] == self.encodedPythonString[0], '__getitem__() fails'

    def testRepr(self):
        assert 'abc' in repr(univ.OctetString('abc'))

    def testAsOctets(self):
        assert univ.OctetString(self.encodedPythonString).asOctets() == self.encodedPythonString, 'testAsOctets() fails'

    def testAsInts(self):
        assert univ.OctetString(self.encodedPythonString).asNumbers() == self.numbersString, 'testAsNumbers() fails'

    def testAdd(self):
        assert univ.OctetString(self.encodedPythonString) + self.encodedPythonString == self.encodedPythonString + self.encodedPythonString, '__add__() fails'

    def testRadd(self):
        assert self.encodedPythonString + univ.OctetString(self.encodedPythonString) == self.encodedPythonString + self.encodedPythonString, '__radd__() fails'

    def testMul(self):
        assert univ.OctetString(self.encodedPythonString) * 2 == self.encodedPythonString * 2, '__mul__() fails'

    def testRmul(self):
        assert 2 * univ.OctetString(self.encodedPythonString) == 2 * self.encodedPythonString, '__rmul__() fails'

    def testContains(self):
        s = univ.OctetString(self.encodedPythonString)
        assert self.encodedPythonString in s
        assert self.encodedPythonString * 2 not in s
    if sys.version_info[:2] > (2, 4):

        def testReverse(self):
            assert list(reversed(univ.OctetString(self.encodedPythonString))) == list(reversed(self.encodedPythonString))