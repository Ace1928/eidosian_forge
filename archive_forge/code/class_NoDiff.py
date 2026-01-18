class NoDiff(BzrError):
    _fmt = 'Diff is not installed on this machine: %(msg)s'

    def __init__(self, msg):
        BzrError.__init__(self, msg=msg)