from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ExprNode(object):

    def __init__(self, tokens):
        self.tokens = tokens[0]

    def eval(self):
        return None