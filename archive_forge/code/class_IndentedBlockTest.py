from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class IndentedBlockTest(ParseTestCase):

    def runTest(self):
        import textwrap
        EQ = pp.Suppress('=')
        stack = [1]
        key = pp.pyparsing_common.identifier
        value = pp.Forward()
        key_value = key + EQ + value
        compound_value = pp.Dict(pp.ungroup(pp.indentedBlock(key_value, stack)))
        value <<= pp.pyparsing_common.integer | pp.QuotedString("'") | compound_value
        parser = pp.Dict(pp.OneOrMore(pp.Group(key_value)))
        text = "\n            a = 100\n            b = 101\n            c =\n                c1 = 200\n                c2 =\n                    c21 = 999\n                c3 = 'A horse, a horse, my kingdom for a horse'\n            d = 505\n        "
        text = textwrap.dedent(text)
        print_(text)
        result = parser.parseString(text)
        print_(result.dump())
        self.assertEqual(result.a, 100, 'invalid indented block result')
        self.assertEqual(result.c.c1, 200, 'invalid indented block result')
        self.assertEqual(result.c.c2.c21, 999, 'invalid indented block result')