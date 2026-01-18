from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PickleTest_Greeting:

    def __init__(self, toks):
        self.salutation = toks[0]
        self.greetee = toks[1]

    def __repr__(self):
        return '%s: {%s}' % (self.__class__.__name__, ', '.join(('%r: %r' % (k, getattr(self, k)) for k in sorted(self.__dict__))))