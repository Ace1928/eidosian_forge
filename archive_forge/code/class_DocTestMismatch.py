import doctest
import re
from ._impl import Mismatch
class DocTestMismatch(Mismatch):
    """Mismatch object for DocTestMatches."""

    def __init__(self, matcher, with_nl):
        self.matcher = matcher
        self.with_nl = with_nl

    def describe(self):
        return self.matcher._describe_difference(self.with_nl)