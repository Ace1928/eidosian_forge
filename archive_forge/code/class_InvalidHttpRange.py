class InvalidHttpRange(InvalidHttpResponse):
    _fmt = 'Invalid http range %(range)r for %(path)s: %(msg)s'

    def __init__(self, path, range, msg):
        self.range = range
        InvalidHttpResponse.__init__(self, path, msg)