import random
from io import StringIO
import struct
import dns.exception
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
from ._compat import string_types
class IncompatibleTypes(dns.exception.DNSException):
    """An attempt was made to add DNS RR data of an incompatible type."""