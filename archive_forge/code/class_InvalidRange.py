class InvalidRange(TransportError):
    _fmt = 'Invalid range access in %(path)s at %(offset)s: %(msg)s'

    def __init__(self, path, offset, msg=None):
        TransportError.__init__(self, msg)
        self.path = path
        self.offset = offset