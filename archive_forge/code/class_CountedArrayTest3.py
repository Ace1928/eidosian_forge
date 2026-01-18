from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CountedArrayTest3(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, nums, OneOrMore, countedArray, alphas
        int_chars = '_' + alphas
        array_counter = Word(int_chars).setParseAction(lambda t: int_chars.index(t[0]))
        testString = 'B 5 7 F 0 1 2 3 4 5 _ C 5 4 3'
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        countedField = countedArray(integer, intExpr=array_counter)
        r = OneOrMore(countedField).parseString(testString)
        print_(testString)
        print_(r.asList())
        self.assertEqual(r.asList(), [[5, 7], [0, 1, 2, 3, 4, 5], [], [5, 4, 3]], 'Failed matching countedArray, got ' + str(r.asList()))