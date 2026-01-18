from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
class UnknownHeaderField(dns.exception.DNSException):
    """The header field name was not recognized when converting from text
    into a message."""