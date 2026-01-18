class BoundBranchConnectionFailure(BzrError):
    _fmt = 'Unable to connect to target of bound branch %(branch)s => %(target)s: %(error)s'

    def __init__(self, branch, target, error):
        BzrError.__init__(self)
        self.branch = branch
        self.target = target
        self.error = error