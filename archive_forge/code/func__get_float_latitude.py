from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _get_float_latitude(self):
    return _tuple_to_float(self.latitude)