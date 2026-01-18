from twisted.internet.defer import TimeoutError
class DNSUnknownError(DomainError):
    """
    Indicates a query failed with an unknown result.
    """