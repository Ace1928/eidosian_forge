class NoCommonAncestor(BzrError):
    _fmt = 'Revisions have no common ancestor: %(revision_a)s %(revision_b)s'

    def __init__(self, revision_a, revision_b):
        self.revision_a = revision_a
        self.revision_b = revision_b