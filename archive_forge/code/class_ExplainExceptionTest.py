from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ExplainExceptionTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        expr = pp.Word(pp.nums).setName('int') + pp.Word(pp.alphas).setName('word')
        try:
            expr.parseString('123 355')
        except pp.ParseException as pe:
            exc = pe
            if not hasattr(exc, '__traceback__'):
                etype, value, traceback = sys.exc_info()
                exc.__traceback__ = traceback
            print_(pp.ParseException.explain(pe, depth=0))
        expr = pp.Word(pp.nums).setName('int') - pp.Word(pp.alphas).setName('word')
        try:
            expr.parseString('123 355 (test using ErrorStop)')
        except pp.ParseSyntaxException as pe:
            exc = pe
            if not hasattr(exc, '__traceback__'):
                etype, value, traceback = sys.exc_info()
                exc.__traceback__ = traceback
            print_(pp.ParseException.explain(pe))
        integer = pp.Word(pp.nums).setName('int').addParseAction(lambda t: int(t[0]))
        expr = integer + integer

        def divide_args(t):
            integer.parseString('A')
            return t[0] / t[1]
        expr.addParseAction(divide_args)
        pp.ParserElement.enablePackrat()
        print_()
        try:
            expr.parseString('123 0')
        except pp.ParseException as pe:
            exc = pe
            if not hasattr(exc, '__traceback__'):
                etype, value, traceback = sys.exc_info()
                exc.__traceback__ = traceback
            print_(pp.ParseException.explain(pe))
        except Exception as exc:
            if not hasattr(exc, '__traceback__'):
                etype, value, traceback = sys.exc_info()
                exc.__traceback__ = traceback
            print_(pp.ParseException.explain(exc))
            raise