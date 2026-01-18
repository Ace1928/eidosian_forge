class InvalidRevisionNumber(BzrError):
    _fmt = 'Invalid revision number %(revno)s'

    def __init__(self, revno):
        BzrError.__init__(self)
        self.revno = revno