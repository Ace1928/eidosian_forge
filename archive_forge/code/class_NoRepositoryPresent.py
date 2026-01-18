class NoRepositoryPresent(BzrError):
    _fmt = 'No repository present: "%(path)s"'

    def __init__(self, controldir):
        BzrError.__init__(self)
        self.path = controldir.transport.clone('..').base