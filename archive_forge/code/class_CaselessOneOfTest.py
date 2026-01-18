from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CaselessOneOfTest(ParseTestCase):

    def runTest(self):
        from pyparsing import oneOf, ZeroOrMore
        caseless1 = oneOf('d a b c aA B A C', caseless=True)
        caseless1str = str(caseless1)
        print_(caseless1str)
        caseless2 = oneOf('d a b c Aa B A C', caseless=True)
        caseless2str = str(caseless2)
        print_(caseless2str)
        self.assertEqual(caseless1str.upper(), caseless2str.upper(), 'oneOf not handling caseless option properly')
        self.assertNotEqual(caseless1str, caseless2str, 'Caseless option properly sorted')
        res = ZeroOrMore(caseless1).parseString('AAaaAaaA')
        print_(res)
        self.assertEqual(len(res), 4, 'caseless1 oneOf failed')
        self.assertEqual(''.join(res), 'aA' * 4, 'caseless1 CaselessLiteral return failed')
        res = ZeroOrMore(caseless2).parseString('AAaaAaaA')
        print_(res)
        self.assertEqual(len(res), 4, 'caseless2 oneOf failed')
        self.assertEqual(''.join(res), 'Aa' * 4, 'caseless1 CaselessLiteral return failed')