class InconsistentDelta(BzrError):
    """Used when we get a delta that is not valid."""
    _fmt = 'An inconsistent delta was supplied involving %(path)r, %(file_id)r\nreason: %(reason)s'

    def __init__(self, path, file_id, reason):
        BzrError.__init__(self)
        self.path = path
        self.file_id = file_id
        self.reason = reason