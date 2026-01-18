from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ChainedTernaryOperator(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        TERNARY_INFIX = pp.infixNotation(pp.pyparsing_common.integer, [(('?', ':'), 3, pp.opAssoc.LEFT)])
        self.assertParseAndCheckList(TERNARY_INFIX, '1?1:0?1:0', [[1, '?', 1, ':', 0, '?', 1, ':', 0]])
        TERNARY_INFIX = pp.infixNotation(pp.pyparsing_common.integer, [(('?', ':'), 3, pp.opAssoc.RIGHT)])
        self.assertParseAndCheckList(TERNARY_INFIX, '1?1:0?1:0', [[1, '?', 1, ':', [0, '?', 1, ':', 0]]])