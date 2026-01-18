class RecursiveBind(BzrError):
    _fmt = 'Branch "%(branch_url)s" appears to be bound to itself. Please use `brz unbind` to fix.'

    def __init__(self, branch_url):
        self.branch_url = branch_url