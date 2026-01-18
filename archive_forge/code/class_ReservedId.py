class ReservedId(BzrError):
    _fmt = 'Reserved revision-id {%(revision_id)s}'

    def __init__(self, revision_id):
        self.revision_id = revision_id