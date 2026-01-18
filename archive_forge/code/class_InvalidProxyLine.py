class InvalidProxyLine(ParseException):

    def __init__(self, line):
        self.line = line
        self.code = 400

    def __str__(self):
        return 'Invalid PROXY line: %r' % self.line