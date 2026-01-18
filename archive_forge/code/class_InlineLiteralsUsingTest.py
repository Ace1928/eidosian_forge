from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InlineLiteralsUsingTest(ParseTestCase):

    def runTest(self):
        from pyparsing import ParserElement, Suppress, Literal, CaselessLiteral, Word, alphas, oneOf, CaselessKeyword, nums
        with AutoReset(ParserElement, '_literalStringClass'):
            ParserElement.inlineLiteralsUsing(Suppress)
            wd = Word(alphas)
            result = (wd + ',' + wd + oneOf('! . ?')).parseString('Hello, World!')
            self.assertEqual(len(result), 3, 'inlineLiteralsUsing(Suppress) failed!')
            ParserElement.inlineLiteralsUsing(Literal)
            result = (wd + ',' + wd + oneOf('! . ?')).parseString('Hello, World!')
            self.assertEqual(len(result), 4, 'inlineLiteralsUsing(Literal) failed!')
            ParserElement.inlineLiteralsUsing(CaselessKeyword)
            result = ('SELECT' + wd + 'FROM' + wd).parseString('select color from colors')
            self.assertEqual(result.asList(), 'SELECT color FROM colors'.split(), 'inlineLiteralsUsing(CaselessKeyword) failed!')
            ParserElement.inlineLiteralsUsing(CaselessLiteral)
            result = ('SELECT' + wd + 'FROM' + wd).parseString('select color from colors')
            self.assertEqual(result.asList(), 'SELECT color FROM colors'.split(), 'inlineLiteralsUsing(CaselessLiteral) failed!')
            integer = Word(nums)
            ParserElement.inlineLiteralsUsing(Literal)
            date_str = integer('year') + '/' + integer('month') + '/' + integer('day')
            result = date_str.parseString('1999/12/31')
            self.assertEqual(result.asList(), ['1999', '/', '12', '/', '31'], 'inlineLiteralsUsing(example 1) failed!')
            ParserElement.inlineLiteralsUsing(Suppress)
            date_str = integer('year') + '/' + integer('month') + '/' + integer('day')
            result = date_str.parseString('1999/12/31')
            self.assertEqual(result.asList(), ['1999', '12', '31'], 'inlineLiteralsUsing(example 2) failed!')