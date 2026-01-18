from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CallableS0(object):

    def __call__():
        return
    __call__ = staticmethod(__call__)