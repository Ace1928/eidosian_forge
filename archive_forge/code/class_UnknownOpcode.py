import dns.exception
class UnknownOpcode(dns.exception.DNSException):
    """An DNS opcode is unknown."""