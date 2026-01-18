import struct
import dns.exception
import dns.rdata
import dns.rdatatype
import dns.name
from dns._compat import xrange
CSYNC record

    @ivar serial: the SOA serial number
    @type serial: int
    @ivar flags: the CSYNC flags
    @type flags: int
    @ivar windows: the windowed bitmap list
    @type windows: list of (window number, string) tuples