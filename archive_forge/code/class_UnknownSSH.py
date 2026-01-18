class UnknownSSH(BzrError):
    _fmt = 'Unrecognised value for BRZ_SSH environment variable: %(vendor)s'

    def __init__(self, vendor):
        BzrError.__init__(self)
        self.vendor = vendor