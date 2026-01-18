from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def _is_rsa(algorithm):
    return algorithm in (RSAMD5, RSASHA1, RSASHA1NSEC3SHA1, RSASHA256, RSASHA512)