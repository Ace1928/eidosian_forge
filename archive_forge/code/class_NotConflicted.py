class NotConflicted(BzrError):
    _fmt = 'File %(filename)s is not conflicted.'

    def __init__(self, filename):
        BzrError.__init__(self)
        self.filename = filename