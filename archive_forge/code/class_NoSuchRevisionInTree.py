class NoSuchRevisionInTree(NoSuchRevision):
    """When using Tree.revision_tree, and the revision is not accessible."""
    _fmt = 'The revision id {%(revision_id)s} is not present in the tree %(tree)s.'

    def __init__(self, tree, revision_id):
        BzrError.__init__(self)
        self.tree = tree
        self.revision_id = revision_id