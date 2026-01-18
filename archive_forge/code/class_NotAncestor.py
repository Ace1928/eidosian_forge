class NotAncestor(BzrError):
    _fmt = 'Revision %(rev_id)s is not an ancestor of %(not_ancestor_id)s'

    def __init__(self, rev_id, not_ancestor_id):
        BzrError.__init__(self, rev_id=rev_id, not_ancestor_id=not_ancestor_id)