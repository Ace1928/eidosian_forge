class InvalidRequestMethod(ParseException):

    def __init__(self, method):
        self.method = method

    def __str__(self):
        return 'Invalid HTTP method: %r' % self.method