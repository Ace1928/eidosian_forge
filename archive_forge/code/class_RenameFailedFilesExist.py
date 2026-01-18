class RenameFailedFilesExist(BzrError):
    """Used when renaming and both source and dest exist."""
    _fmt = 'Could not rename %(source)s => %(dest)s because both files exist. (Use --after to tell brz about a rename that has already happened)%(extra)s'

    def __init__(self, source, dest, extra=None):
        BzrError.__init__(self)
        self.source = str(source)
        self.dest = str(dest)
        if extra:
            self.extra = ' ' + str(extra)
        else:
            self.extra = ''