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
def _is_sha1(algorithm):
    return algorithm in (DSA, RSASHA1, DSANSEC3SHA1, RSASHA1NSEC3SHA1)