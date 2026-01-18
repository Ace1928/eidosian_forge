from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InfixNotationGrammarTest4(ParseTestCase):

    def runTest(self):
        word = pp.Word(pp.alphas)

        def supLiteral(s):
            """Returns the suppressed literal s"""
            return pp.Literal(s).suppress()

        def booleanExpr(atom):
            ops = [(supLiteral('!'), 1, pp.opAssoc.RIGHT, lambda s, l, t: ['!', t[0][0]]), (pp.oneOf('= !='), 2, pp.opAssoc.LEFT), (supLiteral('&'), 2, pp.opAssoc.LEFT, lambda s, l, t: ['&', t[0]]), (supLiteral('|'), 2, pp.opAssoc.LEFT, lambda s, l, t: ['|', t[0]])]
            return pp.infixNotation(atom, ops)
        f = booleanExpr(word) + pp.StringEnd()
        tests = [('bar = foo', "[['bar', '=', 'foo']]"), ('bar = foo & baz = fee', "['&', [['bar', '=', 'foo'], ['baz', '=', 'fee']]]")]
        for test, expected in tests:
            print_(test)
            results = f.parseString(test)
            print_(results)
            self.assertEqual(str(results), expected, "failed to match expected results, got '%s'" % str(results))
            print_()