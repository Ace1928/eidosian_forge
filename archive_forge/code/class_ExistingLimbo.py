class ExistingLimbo(BzrError):
    _fmt = 'This tree contains left-over files from a failed operation.\n    Please examine %(limbo_dir)s to see if it contains any files you wish to\n    keep, and delete it when you are done.'

    def __init__(self, limbo_dir):
        BzrError.__init__(self)
        self.limbo_dir = limbo_dir