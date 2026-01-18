class PathError(BzrError):
    _fmt = 'Generic path error: %(path)r%(extra)s)'

    def __init__(self, path, extra=None):
        BzrError.__init__(self)
        self.path = path
        if extra:
            self.extra = ': ' + str(extra)
        else:
            self.extra = ''