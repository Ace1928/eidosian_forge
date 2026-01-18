import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _pushFirst(str, loc, toks):
    if debug_flag:
        print('pushing ', toks[0], 'str is ', str)
    exprStack.append(toks[0])