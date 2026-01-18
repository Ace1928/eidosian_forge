from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InfixNotationGrammarTest5(ParseTestCase):

    def runTest(self):
        from pyparsing import infixNotation, opAssoc, pyparsing_common as ppc, Literal, oneOf
        expop = Literal('**')
        signop = oneOf('+ -')
        multop = oneOf('* /')
        plusop = oneOf('+ -')

        class ExprNode(object):

            def __init__(self, tokens):
                self.tokens = tokens[0]

            def eval(self):
                return None

        class NumberNode(ExprNode):

            def eval(self):
                return self.tokens

        class SignOp(ExprNode):

            def eval(self):
                mult = {'+': 1, '-': -1}[self.tokens[0]]
                return mult * self.tokens[1].eval()

        class BinOp(ExprNode):

            def eval(self):
                ret = self.tokens[0].eval()
                for op, operand in zip(self.tokens[1::2], self.tokens[2::2]):
                    ret = self.opn_map[op](ret, operand.eval())
                return ret

        class ExpOp(BinOp):
            opn_map = {'**': lambda a, b: b ** a}

        class MultOp(BinOp):
            import operator
            opn_map = {'*': operator.mul, '/': operator.truediv}

        class AddOp(BinOp):
            import operator
            opn_map = {'+': operator.add, '-': operator.sub}
        operand = ppc.number().setParseAction(NumberNode)
        expr = infixNotation(operand, [(expop, 2, opAssoc.LEFT, (lambda pr: [pr[0][::-1]], ExpOp)), (signop, 1, opAssoc.RIGHT, SignOp), (multop, 2, opAssoc.LEFT, MultOp), (plusop, 2, opAssoc.LEFT, AddOp)])
        tests = '            2+7\n            2**3\n            2**3**2\n            3**9\n            3**3**2\n            '
        for t in tests.splitlines():
            t = t.strip()
            if not t:
                continue
            parsed = expr.parseString(t)
            eval_value = parsed[0].eval()
            self.assertEqual(eval_value, eval(t), 'Error evaluating %r, expected %r, got %r' % (t, eval(t), eval_value))