class BranchError(BzrError):
    """Base class for concrete 'errors about a branch'."""

    def __init__(self, branch):
        BzrError.__init__(self, branch=branch)