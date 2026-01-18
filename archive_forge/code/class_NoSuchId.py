class NoSuchId(BzrError):
    _fmt = 'The file id "%(file_id)s" is not present in the tree %(tree)s.'

    def __init__(self, tree, file_id):
        BzrError.__init__(self)
        self.file_id = file_id
        self.tree = tree