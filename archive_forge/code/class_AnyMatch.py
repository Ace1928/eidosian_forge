import types
from ._impl import (
class AnyMatch:
    """Matches if any of the provided values match the given matcher."""

    def __init__(self, matcher):
        self.matcher = matcher

    def __str__(self):
        return f'AnyMatch({self.matcher})'

    def match(self, values):
        mismatches = []
        for value in values:
            mismatch = self.matcher.match(value)
            if mismatch:
                mismatches.append(mismatch)
            else:
                return None
        return MismatchesAll(mismatches)