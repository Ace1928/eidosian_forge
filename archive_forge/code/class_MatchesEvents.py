import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
class MatchesEvents:
    """Match a list of test result events.

    Specify events as a data structure.  Ordinary Python objects within this
    structure will be compared exactly, but you can also use matchers at any
    point.
    """

    def __init__(self, *expected):
        self._expected = expected

    def _make_matcher(self, obj):
        if hasattr(obj, 'match'):
            return obj
        elif isinstance(obj, tuple) or isinstance(obj, list):
            return MatchesListwise([self._make_matcher(item) for item in obj])
        elif isinstance(obj, dict):
            return MatchesDict({key: self._make_matcher(value) for key, value in obj.items()})
        else:
            return Equals(obj)

    def match(self, observed):
        matcher = self._make_matcher(self._expected)
        return matcher.match(observed)