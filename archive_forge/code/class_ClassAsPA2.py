from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ClassAsPA2(object):

    def __init__(self, l, t):
        self.t = t

    def __str__(self):
        return self.t[0]