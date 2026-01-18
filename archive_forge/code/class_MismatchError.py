from testtools.compat import (
class MismatchError(AssertionError):
    """Raised when a mismatch occurs."""

    def __init__(self, matchee, matcher, mismatch, verbose=False):
        super().__init__()
        self.matchee = matchee
        self.matcher = matcher
        self.mismatch = mismatch
        self.verbose = verbose

    def __str__(self):
        difference = self.mismatch.describe()
        if self.verbose:
            if isinstance(self.matchee, (str, bytes)):
                matchee = text_repr(self.matchee, multiline=False)
            else:
                matchee = repr(self.matchee)
            return 'Match failed. Matchee: %s\nMatcher: %s\nDifference: %s\n' % (matchee, self.matcher, difference)
        else:
            return difference