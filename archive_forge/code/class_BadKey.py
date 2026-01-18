import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
class BadKey(dns.exception.DNSException):
    """The TSIG record owner name does not match the key."""