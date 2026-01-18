class UpToDateFormat(BzrError):
    _fmt = 'The branch format %(format)s is already at the most recent format.'

    def __init__(self, format):
        BzrError.__init__(self)
        self.format = format