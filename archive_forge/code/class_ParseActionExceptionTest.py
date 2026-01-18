from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseActionExceptionTest(ParseTestCase):

    def runTest(self):
        self.expect_traceback = True
        import pyparsing as pp
        import traceback
        number = pp.Word(pp.nums)

        def number_action():
            raise IndexError
        number.setParseAction(number_action)
        symbol = pp.Word('abcd', max=1)
        expr = number | symbol
        try:
            expr.parseString('1 + 2')
        except Exception as e:
            self.assertTrue(hasattr(e, '__cause__'), 'no __cause__ attribute in the raised exception')
            self.assertTrue(e.__cause__ is not None, '__cause__ not propagated to outer exception')
            self.assertTrue(type(e.__cause__) == IndexError, '__cause__ references wrong exception')
            traceback.print_exc()
        else:
            self.assertTrue(False, 'Expected ParseException not raised')