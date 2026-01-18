from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesAllDictInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesAllDict({'a': NotEquals(1), 'b': NotEquals(2)})
    matches_matches = [3, 4]
    matches_mismatches = [1, 2]
    str_examples = [("MatchesAllDict({'a': NotEquals(1), 'b': NotEquals(2)})", matches_matcher)]
    describe_examples = [('a: 1 == 1', 1, matches_matcher)]