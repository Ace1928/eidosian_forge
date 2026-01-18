from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsDelTest(ParseTestCase):

    def runTest(self):
        from pyparsing import OneOrMore, Word, alphas, nums
        grammar = OneOrMore(Word(nums))('ints') + OneOrMore(Word(alphas))('words')
        res = grammar.parseString('123 456 ABC DEF')
        print_(res.dump())
        origInts = res.ints.asList()
        origWords = res.words.asList()
        del res[1]
        del res['words']
        print_(res.dump())
        self.assertEqual(res[1], 'ABC', "failed to delete 0'th element correctly")
        self.assertEqual(res.ints.asList(), origInts, 'updated named attributes, should have updated list only')
        self.assertEqual(res.words, '', 'failed to update named attribute correctly')
        self.assertEqual(res[-1], 'DEF', 'updated list, should have updated named attributes only')