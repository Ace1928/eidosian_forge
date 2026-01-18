import types
from ._impl import (
class MatchesAll:
    """Matches if all of the matchers it is created with match."""

    def __init__(self, *matchers, **options):
        """Construct a MatchesAll matcher.

        Just list the component matchers as arguments in the ``*args``
        style. If you want only the first mismatch to be reported, past in
        first_only=True as a keyword argument. By default, all mismatches are
        reported.
        """
        self.matchers = matchers
        self.first_only = options.get('first_only', False)

    def __str__(self):
        return 'MatchesAll(%s)' % ', '.join(map(str, self.matchers))

    def match(self, matchee):
        results = []
        for matcher in self.matchers:
            mismatch = matcher.match(matchee)
            if mismatch is not None:
                if self.first_only:
                    return mismatch
                results.append(mismatch)
        if results:
            return MismatchesAll(results)
        else:
            return None