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
class NoValueTestCase(BaseTestCase):

    def testSingleton(self):
        assert univ.NoValue() is univ.NoValue(), 'NoValue is not a singleton'

    def testRepr(self):
        try:
            repr(univ.noValue)
        except PyAsn1Error:
            assert False, 'repr() on NoValue object fails'

    def testIsInstance(self):
        try:
            assert isinstance(univ.noValue, univ.NoValue), 'isinstance() on NoValue() object fails'
        except PyAsn1Error:
            assert False, 'isinstance() on NoValue object fails'

    def testStr(self):
        try:
            str(univ.noValue)
        except PyAsn1Error:
            pass
        else:
            assert False, 'str() works for NoValue object'

    def testLen(self):
        try:
            len(univ.noValue)
        except PyAsn1Error:
            pass
        else:
            assert False, 'len() works for NoValue object'

    def testCmp(self):
        try:
            univ.noValue == 1
        except PyAsn1Error:
            pass
        else:
            assert False, 'comparison works for NoValue object'

    def testSubs(self):
        try:
            univ.noValue[0]
        except PyAsn1Error:
            pass
        else:
            assert False, '__getitem__() works for NoValue object'

    def testKey(self):
        try:
            univ.noValue['key']
        except PyAsn1Error:
            pass
        else:
            assert False, '__getitem__() works for NoValue object'

    def testKeyAssignment(self):
        try:
            univ.noValue['key'] = 123
        except PyAsn1Error:
            pass
        else:
            assert False, '__setitem__() works for NoValue object'

    def testInt(self):
        try:
            int(univ.noValue)
        except PyAsn1Error:
            pass
        else:
            assert False, 'integer conversion works for NoValue object'

    def testAdd(self):
        try:
            univ.noValue + univ.noValue
        except PyAsn1Error:
            pass
        else:
            assert False, 'addition works for NoValue object'

    def testBitShift(self):
        try:
            univ.noValue << 1
        except PyAsn1Error:
            pass
        else:
            assert False, 'bitshift works for NoValue object'

    def testBooleanEvaluation(self):
        try:
            if univ.noValue:
                pass
        except PyAsn1Error:
            pass
        else:
            assert False, 'boolean evaluation works for NoValue object'

    def testSizeOf(self):
        try:
            if hasattr(sys, 'getsizeof'):
                sys.getsizeof(univ.noValue)
        except PyAsn1Error:
            assert False, 'sizeof failed for NoValue object'