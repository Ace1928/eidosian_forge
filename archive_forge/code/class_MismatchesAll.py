import types
from ._impl import (
class MismatchesAll(Mismatch):
    """A mismatch with many child mismatches."""

    def __init__(self, mismatches, wrap=True):
        self.mismatches = mismatches
        self._wrap = wrap

    def describe(self):
        descriptions = []
        if self._wrap:
            descriptions = ['Differences: [']
        for mismatch in self.mismatches:
            descriptions.append(mismatch.describe())
        if self._wrap:
            descriptions.append(']')
        return '\n'.join(descriptions)