class NotWriteLocked(BzrError):
    _fmt = '%(not_locked)r is not write locked but needs to be.'

    def __init__(self, not_locked):
        self.not_locked = not_locked