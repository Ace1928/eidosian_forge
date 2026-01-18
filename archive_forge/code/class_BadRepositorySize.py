class BadRepositorySize(ImportError):
    """Raised when the repository has an incorrect number of revisions."""
    _fmt = 'Bad repository size - %(found)d revisions found, %(expected)d expected'

    def __init__(self, expected, found):
        self.expected = expected
        self.found = found
        ImportError.__init__(self)