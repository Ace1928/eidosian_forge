from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class LineAndStringEndTest(ParseTestCase):

    def runTest(self):
        from pyparsing import OneOrMore, lineEnd, alphanums, Word, stringEnd, delimitedList, SkipTo
        NLs = OneOrMore(lineEnd)
        bnf1 = delimitedList(Word(alphanums).leaveWhitespace(), NLs)
        bnf2 = Word(alphanums) + stringEnd
        bnf3 = Word(alphanums) + SkipTo(stringEnd)
        tests = [('testA\ntestB\ntestC\n', ['testA', 'testB', 'testC']), ('testD\ntestE\ntestF', ['testD', 'testE', 'testF']), ('a', ['a'])]
        for test, expected in tests:
            res1 = bnf1.parseString(test)
            print_(res1, '=?', expected)
            self.assertEqual(res1.asList(), expected, 'Failed lineEnd/stringEnd test (1): ' + repr(test) + ' -> ' + str(res1.asList()))
            res2 = bnf2.searchString(test)[0]
            print_(res2.asList(), '=?', expected[-1:])
            self.assertEqual(res2.asList(), expected[-1:], 'Failed lineEnd/stringEnd test (2): ' + repr(test) + ' -> ' + str(res2.asList()))
            res3 = bnf3.parseString(test)
            first = res3[0]
            rest = res3[1]
            print_(repr(rest), '=?', repr(test[len(first) + 1:]))
            self.assertEqual(rest, test[len(first) + 1:], 'Failed lineEnd/stringEnd test (3): ' + repr(test) + ' -> ' + str(res3.asList()))
            print_()
        from pyparsing import Regex
        import re
        k = Regex('a+', flags=re.S + re.M)
        k = k.parseWithTabs()
        k = k.leaveWhitespace()
        tests = [('aaa', ['aaa']), ('\\naaa', None), ('a\\naa', None), ('aaa\\n', None)]
        for i, (src, expected) in enumerate(tests):
            print_(i, repr(src).replace('\\\\', '\\'), end=' ')
            try:
                res = k.parseString(src, parseAll=True).asList()
            except ParseException as pe:
                res = None
            print_(res)
            self.assertEqual(res, expected, 'Failed on parseAll=True test %d' % i)