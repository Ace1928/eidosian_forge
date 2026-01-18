from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseKeywordTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, Keyword
        kw = Keyword('if')
        lit = Literal('if')

        def test(s, litShouldPass, kwShouldPass):
            print_('Test', s)
            print_('Match Literal', end=' ')
            try:
                print_(lit.parseString(s))
            except Exception:
                print_('failed')
                if litShouldPass:
                    self.assertTrue(False, 'Literal failed to match %s, should have' % s)
            else:
                if not litShouldPass:
                    self.assertTrue(False, 'Literal matched %s, should not have' % s)
            print_('Match Keyword', end=' ')
            try:
                print_(kw.parseString(s))
            except Exception:
                print_('failed')
                if kwShouldPass:
                    self.assertTrue(False, 'Keyword failed to match %s, should have' % s)
            else:
                if not kwShouldPass:
                    self.assertTrue(False, 'Keyword matched %s, should not have' % s)
        test('ifOnlyIfOnly', True, False)
        test('if(OnlyIfOnly)', True, True)
        test('if (OnlyIf Only)', True, True)
        kw = Keyword('if', caseless=True)
        test('IFOnlyIfOnly', False, False)
        test('If(OnlyIfOnly)', False, True)
        test('iF (OnlyIf Only)', False, True)