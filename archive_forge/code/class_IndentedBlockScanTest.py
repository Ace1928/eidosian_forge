from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class IndentedBlockScanTest(ParseTestCase):

    def get_parser(self):
        """
        A valid statement is the word "block:", followed by an indent, followed by the letter A only, or another block
        """
        stack = [1]
        block = pp.Forward()
        body = pp.indentedBlock(pp.Literal('A') ^ block, indentStack=stack, indent=True)
        block <<= pp.Literal('block:') + body
        return block

    def runTest(self):
        from textwrap import dedent
        p1 = self.get_parser()
        r1 = list(p1.scanString(dedent('        block:\n            A\n        ')))
        self.assertEqual(len(r1), 1)
        p2 = self.get_parser()
        r2 = list(p2.scanString(dedent('        block:\n            B\n        ')))
        self.assertEqual(len(r2), 0)
        p3 = self.get_parser()
        r3 = list(p3.scanString(dedent('        block:\n            A\n        block:\n            B\n        ')))
        self.assertEqual(len(r3), 1)
        p4 = self.get_parser()
        r4 = list(p4.scanString(dedent('        block:\n            B\n        block:\n            A\n        ')))
        self.assertEqual(len(r4), 1)
        p5 = self.get_parser()
        r5 = list(p5.scanString(dedent('        block:\n            block:\n                A\n        block:\n            block:\n                B\n        ')))
        self.assertEqual(len(r5), 1)
        p6 = self.get_parser()
        r6 = list(p6.scanString(dedent('        block:\n            block:\n                B\n        block:\n            block:\n                A\n        ')))
        self.assertEqual(len(r6), 1)