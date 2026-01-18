from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class LiteralExceptionTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        for cls in (pp.Literal, pp.CaselessLiteral, pp.Keyword, pp.CaselessKeyword, pp.Word, pp.Regex):
            expr = cls('xyz')
            try:
                expr.parseString(' ')
            except Exception as e:
                print_(cls.__name__, str(e))
                self.assertTrue(isinstance(e, pp.ParseBaseException), 'class {0} raised wrong exception type {1}'.format(cls.__name__, type(e).__name__))