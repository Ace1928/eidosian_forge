class WorkingTreeNotRevision(BzrError):
    _fmt = 'The working tree for %(basedir)s has changed since the last commit, but weave merge requires that it be unchanged'

    def __init__(self, tree):
        BzrError.__init__(self, basedir=tree.basedir)