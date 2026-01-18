class UninitializableFormat(BzrError):
    _fmt = 'Format %(format)s cannot be initialised by this version of brz.'

    def __init__(self, format):
        BzrError.__init__(self)
        self.format = format