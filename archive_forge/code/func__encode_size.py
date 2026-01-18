from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _encode_size(what, desc):
    what = long(what)
    exponent = _exponent_of(what, desc) & 15
    base = what // pow(10, exponent) & 15
    return base * 16 + exponent