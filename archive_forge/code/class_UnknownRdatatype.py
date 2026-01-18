import re
import dns.exception
class UnknownRdatatype(dns.exception.DNSException):
    """DNS resource record type is unknown."""