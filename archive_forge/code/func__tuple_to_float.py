from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _tuple_to_float(what):
    value = float(what[0])
    value += float(what[1]) / 60.0
    value += float(what[2]) / 3600.0
    value += float(what[3]) / 3600000.0
    return float(what[4]) * value