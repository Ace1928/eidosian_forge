import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _divfunc(a, b):
    if _isscalar(a) and _isscalar(b):
        return '%s/%s' % (a, b)
    else:
        raise TypeError