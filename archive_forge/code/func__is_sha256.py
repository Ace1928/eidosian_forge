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
def _is_sha256(algorithm):
    return algorithm in (RSASHA256, ECDSAP256SHA256)