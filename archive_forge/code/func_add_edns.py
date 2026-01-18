from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def add_edns(self, edns, ednsflags, payload, options=None):
    """Add an EDNS OPT record to the message."""
    ednsflags &= long(4278255615)
    ednsflags |= edns << 16
    self._set_section(ADDITIONAL)
    before = self.output.tell()
    self.output.write(struct.pack('!BHHIH', 0, dns.rdatatype.OPT, payload, ednsflags, 0))
    if options is not None:
        lstart = self.output.tell()
        for opt in options:
            stuff = struct.pack('!HH', opt.otype, 0)
            self.output.write(stuff)
            start = self.output.tell()
            opt.to_wire(self.output)
            end = self.output.tell()
            assert end - start < 65536
            self.output.seek(start - 2)
            stuff = struct.pack('!H', end - start)
            self.output.write(stuff)
            self.output.seek(0, 2)
        lend = self.output.tell()
        assert lend - lstart < 65536
        self.output.seek(lstart - 2)
        stuff = struct.pack('!H', lend - lstart)
        self.output.write(stuff)
        self.output.seek(0, 2)
    after = self.output.tell()
    if after >= self.max_size:
        self._rollback(before)
        raise dns.exception.TooBig
    self.counts[ADDITIONAL] += 1