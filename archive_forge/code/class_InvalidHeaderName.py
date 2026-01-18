class InvalidHeaderName(ParseException):

    def __init__(self, hdr):
        self.hdr = hdr

    def __str__(self):
        return 'Invalid HTTP header name: %r' % self.hdr