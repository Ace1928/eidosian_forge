from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def _write_tsig(self, tsig_rdata, keyname):
    self._set_section(ADDITIONAL)
    before = self.output.tell()
    keyname.to_wire(self.output, self.compress, self.origin)
    self.output.write(struct.pack('!HHIH', dns.rdatatype.TSIG, dns.rdataclass.ANY, 0, 0))
    rdata_start = self.output.tell()
    self.output.write(tsig_rdata)
    after = self.output.tell()
    assert after - rdata_start < 65536
    if after >= self.max_size:
        self._rollback(before)
        raise dns.exception.TooBig
    self.output.seek(rdata_start - 2)
    self.output.write(struct.pack('!H', after - rdata_start))
    self.counts[ADDITIONAL] += 1
    self.output.seek(10)
    self.output.write(struct.pack('!H', self.counts[ADDITIONAL]))
    self.output.seek(0, 2)