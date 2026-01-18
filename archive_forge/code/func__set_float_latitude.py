from __future__ import division
import struct
import dns.exception
import dns.rdata
from dns._compat import long, xrange, round_py2_compat
def _set_float_latitude(self, value):
    self.latitude = _float_to_tuple(value)