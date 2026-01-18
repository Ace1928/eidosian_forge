class RevisionNotPresent(VersionedFileError):
    _fmt = 'Revision {%(revision_id)s} not present in "%(file_id)s".'

    def __init__(self, revision_id, file_id):
        VersionedFileError.__init__(self)
        self.revision_id = revision_id
        self.file_id = file_id