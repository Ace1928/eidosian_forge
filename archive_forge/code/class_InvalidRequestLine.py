class InvalidRequestLine(ParseException):

    def __init__(self, req):
        self.req = req
        self.code = 400

    def __str__(self):
        return 'Invalid HTTP request line: %r' % self.req