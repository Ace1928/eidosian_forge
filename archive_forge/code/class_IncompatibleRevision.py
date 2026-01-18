class IncompatibleRevision(BzrError):
    _fmt = 'Revision is not compatible with %(repo_format)s'

    def __init__(self, repo_format):
        BzrError.__init__(self)
        self.repo_format = repo_format