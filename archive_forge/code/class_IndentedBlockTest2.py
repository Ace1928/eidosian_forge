from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class IndentedBlockTest2(ParseTestCase):

    def runTest(self):
        from textwrap import dedent
        from pyparsing import Word, alphas, alphanums, Suppress, Forward, indentedBlock, Literal, OneOrMore
        indent_stack = [1]
        key = Word(alphas, alphanums) + Suppress(':')
        stmt = Forward()
        suite = indentedBlock(stmt, indent_stack)
        body = key + suite
        pattern = Word(alphas) + Suppress('(') + Word(alphas) + Suppress(')')
        stmt << pattern

        def key_parse_action(toks):
            print_("Parsing '%s'..." % toks[0])
        key.setParseAction(key_parse_action)
        header = Suppress('[') + Literal('test') + Suppress(']')
        content = header + OneOrMore(indentedBlock(body, indent_stack, False))
        contents = Forward()
        suites = indentedBlock(content, indent_stack)
        extra = Literal('extra') + Suppress(':') + suites
        contents << (content | extra)
        parser = OneOrMore(contents)
        sample = dedent('\n        extra:\n            [test]\n            one0: \n                two (three)\n            four0:\n                five (seven)\n        extra:\n            [test]\n            one1: \n                two (three)\n            four1:\n                five (seven)\n        ')
        success, _ = parser.runTests([sample])
        self.assertTrue(success, 'Failed indentedBlock test for issue #87')