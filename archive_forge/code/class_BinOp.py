from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class BinOp(ExprNode):

    def eval(self):
        ret = self.tokens[0].eval()
        for op, operand in zip(self.tokens[1::2], self.tokens[2::2]):
            ret = self.opn_map[op](ret, operand.eval())
        return ret