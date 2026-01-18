class UnstackableRepositoryFormat(BzrError):
    _fmt = "The repository '%(url)s'(%(format)s) is not a stackable format. You will need to upgrade the repository to permit branch stacking."

    def __init__(self, format, url):
        BzrError.__init__(self)
        self.format = format
        self.url = url