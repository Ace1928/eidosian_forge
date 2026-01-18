class DuplicateHelpPrefix(BzrError):
    _fmt = 'The prefix %(prefix)s is in the help search path twice.'

    def __init__(self, prefix):
        self.prefix = prefix