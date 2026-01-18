import random
from io import StringIO
import struct
import dns.exception
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
from ._compat import string_types
class DifferingCovers(dns.exception.DNSException):
    """An attempt was made to add a DNS SIG/RRSIG whose covered type
    is not the same as that of the other rdatas in the rdataset."""