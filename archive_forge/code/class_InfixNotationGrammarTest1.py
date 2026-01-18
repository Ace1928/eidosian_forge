from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InfixNotationGrammarTest1(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, nums, alphas, Literal, oneOf, infixNotation, opAssoc
        import ast
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        variable = Word(alphas, exact=1)
        operand = integer | variable
        expop = Literal('^')
        signop = oneOf('+ -')
        multop = oneOf('* /')
        plusop = oneOf('+ -')
        factop = Literal('!')
        expr = infixNotation(operand, [(factop, 1, opAssoc.LEFT), (expop, 2, opAssoc.RIGHT), (signop, 1, opAssoc.RIGHT), (multop, 2, opAssoc.LEFT), (plusop, 2, opAssoc.LEFT)])
        test = ['9 + 2 + 3', '9 + 2 * 3', '(9 + 2) * 3', '(9 + -2) * 3', '(9 + --2) * 3', '(9 + -2) * 3^2^2', '(9! + -2) * 3^2^2', 'M*X + B', 'M*(X + B)', '1+2*-3^4*5+-+-6', '3!!']
        expected = "[[9, '+', 2, '+', 3]]\n                    [[9, '+', [2, '*', 3]]]\n                    [[[9, '+', 2], '*', 3]]\n                    [[[9, '+', ['-', 2]], '*', 3]]\n                    [[[9, '+', ['-', ['-', 2]]], '*', 3]]\n                    [[[9, '+', ['-', 2]], '*', [3, '^', [2, '^', 2]]]]\n                    [[[[9, '!'], '+', ['-', 2]], '*', [3, '^', [2, '^', 2]]]]\n                    [[['M', '*', 'X'], '+', 'B']]\n                    [['M', '*', ['X', '+', 'B']]]\n                    [[1, '+', [2, '*', ['-', [3, '^', 4]], '*', 5], '+', ['-', ['+', ['-', 6]]]]]\n                    [[3, '!', '!']]".split('\n')
        expected = [ast.literal_eval(x.strip()) for x in expected]
        for t, e in zip(test, expected):
            print_(t, '->', e, 'got', expr.parseString(t).asList())
            self.assertEqual(expr.parseString(t).asList(), e, 'mismatched results for infixNotation: got %s, expected %s' % (expr.parseString(t).asList(), e))