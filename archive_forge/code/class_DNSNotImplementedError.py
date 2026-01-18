from twisted.internet.defer import TimeoutError
class DNSNotImplementedError(DomainError):
    """
    Indicates a query failed with a result of C{twisted.names.dns.ENOTIMP}.
    """