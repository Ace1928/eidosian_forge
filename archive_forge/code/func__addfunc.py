import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def _addfunc(a, b):
    if _isscalar(a) and _isscalar(b):
        return '(%s+%s)' % (a, b)
    if _isvec(a) and _isvec(b):
        return '%svAdd(%s,%s)' % (vprefix, a[vplen:], b[vplen:])
    if _ismat(a) and _ismat(b):
        return '%smAdd(%s,%s)' % (mprefix, a[mplen:], b[mplen:])
    else:
        raise TypeError