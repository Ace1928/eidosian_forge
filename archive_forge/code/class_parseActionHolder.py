from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class parseActionHolder(object):

    def pa3(s, l, t):
        return t
    pa3 = staticmethod(pa3)

    def pa2(l, t):
        return t
    pa2 = staticmethod(pa2)

    def pa1(t):
        return t
    pa1 = staticmethod(pa1)

    def pa0():
        return
    pa0 = staticmethod(pa0)