class NoSuchExportFormat(BzrError):
    _fmt = 'Export format %(format)r not supported'

    def __init__(self, format):
        BzrError.__init__(self)
        self.format = format