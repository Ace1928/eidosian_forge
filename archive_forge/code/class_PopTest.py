from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PopTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, alphas, nums
        source = 'AAA 123 456 789 234'
        patt = Word(alphas)('name') + Word(nums) * (1,)
        result = patt.parseString(source)
        tests = [(0, 'AAA', ['123', '456', '789', '234']), (None, '234', ['123', '456', '789']), ('name', 'AAA', ['123', '456', '789']), (-1, '789', ['123', '456'])]
        for test in tests:
            idx, val, remaining = test
            if idx is not None:
                ret = result.pop(idx)
            else:
                ret = result.pop()
            print_('EXP:', val, remaining)
            print_('GOT:', ret, result.asList())
            print_(ret, result.asList())
            self.assertEqual(ret, val, 'wrong value returned, got %r, expected %r' % (ret, val))
            self.assertEqual(remaining, result.asList(), 'list is in wrong state after pop, got %r, expected %r' % (result.asList(), remaining))
            print_()
        prevlist = result.asList()
        ret = result.pop('name', default='noname')
        print_(ret)
        print_(result.asList())
        self.assertEqual(ret, 'noname', 'default value not successfully returned, got %r, expected %r' % (ret, 'noname'))
        self.assertEqual(result.asList(), prevlist, 'list is in wrong state after pop, got %r, expected %r' % (result.asList(), remaining))