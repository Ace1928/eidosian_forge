import base64
import calendar
import struct
import time
import dns.dnssec
import dns.exception
import dns.rdata
import dns.rdatatype
class BadSigTime(dns.exception.DNSException):
    """Time in DNS SIG or RRSIG resource record cannot be parsed."""