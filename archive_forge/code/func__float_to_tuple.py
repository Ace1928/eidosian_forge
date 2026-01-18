from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _float_to_tuple(what):
    if what < 0:
        sign = -1
        what *= -1
    else:
        sign = 1
    what = round_py2_compat(what * 3600000)
    degrees = int(what // 3600000)
    what -= degrees * 3600000
    minutes = int(what // 60000)
    what -= minutes * 60000
    seconds = int(what // 1000)
    what -= int(seconds * 1000)
    what = int(what)
    return (degrees, minutes, seconds, what, sign)