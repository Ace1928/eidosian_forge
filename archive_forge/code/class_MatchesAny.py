import types
from ._impl import (
class MatchesAny:
    """Matches if any of the matchers it is created with match."""

    def __init__(self, *matchers):
        self.matchers = matchers

    def match(self, matchee):
        results = []
        for matcher in self.matchers:
            mismatch = matcher.match(matchee)
            if mismatch is None:
                return None
            results.append(mismatch)
        return MismatchesAll(results)

    def __str__(self):
        return 'MatchesAny(%s)' % ', '.join([str(matcher) for matcher in self.matchers])