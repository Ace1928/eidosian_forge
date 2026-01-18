import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _evaluateStack(s):
    op = s.pop()
    if op in '+-*/@^':
        op2 = _evaluateStack(s)
        op1 = _evaluateStack(s)
        result = opn[op](op1, op2)
        if debug_flag:
            print(result)
        return result
    else:
        return op