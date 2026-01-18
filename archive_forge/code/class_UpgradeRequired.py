class UpgradeRequired(BzrError):
    _fmt = 'To use this feature you must upgrade your branch at %(path)s.'

    def __init__(self, path):
        BzrError.__init__(self)
        self.path = path