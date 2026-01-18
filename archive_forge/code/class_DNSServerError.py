from twisted.internet.defer import TimeoutError
class DNSServerError(DomainError):
    """
    Indicates a query failed with a result of C{twisted.names.dns.ESERVER}.
    """