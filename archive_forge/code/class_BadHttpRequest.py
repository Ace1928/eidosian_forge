class BadHttpRequest(UnexpectedHttpStatus):
    _fmt = 'Bad http request for %(path)s: %(reason)s'

    def __init__(self, path, reason):
        self.path = path
        self.reason = reason
        TransportError.__init__(self, reason)