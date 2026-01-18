class IncompatibleFormat(BzrError):
    _fmt = 'Format %(format)s is not compatible with .bzr version %(controldir)s.'

    def __init__(self, format, controldir_format):
        BzrError.__init__(self)
        self.format = format
        self.controldir = controldir_format