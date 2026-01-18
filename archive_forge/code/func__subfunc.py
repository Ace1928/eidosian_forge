import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _subfunc(a, b):
    if _isscalar(a) and _isscalar(b):
        return '(%s-%s)' % (a, b)
    if _isvec(a) and _isvec(b):
        return '%svSubtract(%s,%s)' % (vprefix, a[vplen:], b[vplen:])
    if _ismat(a) and _ismat(b):
        return '%smSubtract(%s,%s)' % (mprefix, a[mplen:], b[mplen:])
    else:
        raise TypeError