class NoSuchRevisionSpec(BzrError):
    _fmt = 'No namespace registered for string: %(spec)r'

    def __init__(self, spec):
        BzrError.__init__(self, spec=spec)