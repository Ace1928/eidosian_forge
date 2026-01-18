from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class AddOp(BinOp):
    import operator
    opn_map = {'+': operator.add, '-': operator.sub}