from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _exponent_of(what, desc):
    if what == 0:
        return 0
    exp = None
    for i in xrange(len(_pows)):
        if what // _pows[i] == long(0):
            exp = i - 1
            break
    if exp is None or exp < 0:
        raise dns.exception.SyntaxError('%s value out of bounds' % desc)
    return exp