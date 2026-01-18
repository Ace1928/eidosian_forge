from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class EmptyDictDoesNotRaiseException(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        key = pp.Word(pp.alphas)
        value = pp.Word(pp.nums)
        EQ = pp.Suppress('=')
        key_value_dict = pp.dictOf(key, EQ + value)
        print_(key_value_dict.parseString('            a = 10\n            b = 20\n            ').dump())
        try:
            print_(key_value_dict.parseString('').dump())
        except pp.ParseException as pe:
            exc = pe
            if not hasattr(exc, '__traceback__'):
                etype, value, traceback = sys.exc_info()
                exc.__traceback__ = traceback
            print_(pp.ParseException.explain(pe))
        else:
            self.assertTrue(False, 'failed to raise exception when matching empty string')