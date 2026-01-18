class GhostRevisionsHaveNoRevno(BzrError):
    """When searching for revnos, if we encounter a ghost, we are stuck"""
    _fmt = 'Could not determine revno for {%(revision_id)s} because its ancestry shows a ghost at {%(ghost_revision_id)s}'

    def __init__(self, revision_id, ghost_revision_id):
        self.revision_id = revision_id
        self.ghost_revision_id = ghost_revision_id