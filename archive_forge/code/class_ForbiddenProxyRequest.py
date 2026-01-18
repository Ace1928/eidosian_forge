class ForbiddenProxyRequest(ParseException):

    def __init__(self, host):
        self.host = host
        self.code = 403

    def __str__(self):
        return 'Proxy request from %r not allowed' % self.host