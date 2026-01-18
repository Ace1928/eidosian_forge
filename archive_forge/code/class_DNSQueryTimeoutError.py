from twisted.internet.defer import TimeoutError
class DNSQueryTimeoutError(TimeoutError):
    """
    Indicates a lookup failed due to a timeout.

    @ivar id: The id of the message which timed out.
    """

    def __init__(self, id):
        TimeoutError.__init__(self)
        self.id = id