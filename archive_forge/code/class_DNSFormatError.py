from twisted.internet.defer import TimeoutError
class DNSFormatError(DomainError):
    """
    Indicates a query failed with a result of C{twisted.names.dns.EFORMAT}.
    """