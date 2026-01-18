import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
class BadTime(dns.exception.DNSException):
    """The current time is not within the TSIG's validity time."""