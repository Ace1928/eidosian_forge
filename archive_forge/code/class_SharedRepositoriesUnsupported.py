class SharedRepositoriesUnsupported(UnsupportedOperation):
    _fmt = 'Shared repositories are not supported by %(format)r.'

    def __init__(self, format):
        BzrError.__init__(self, format=format)