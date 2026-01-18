class InvalidBranchName(PathError):
    _fmt = 'Invalid branch name: %(name)s'

    def __init__(self, name):
        BzrError.__init__(self)
        self.name = name