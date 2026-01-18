class IncompatibleBundleFormat(BzrError):
    _fmt = 'Bundle format %(bundle_format)s is incompatible with %(other)s'

    def __init__(self, bundle_format, other):
        BzrError.__init__(self)
        self.bundle_format = bundle_format
        self.other = other