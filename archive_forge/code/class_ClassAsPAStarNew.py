from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ClassAsPAStarNew(tuple):

    def __new__(cls, *args):
        print_('make a ClassAsPAStarNew', args)
        return tuple.__new__(cls, *args[2].asList())

    def __str__(self):
        return ''.join(self)