from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesPredicateWithParams(TestCase, TestMatchersInterface):
    matches_matcher = MatchesPredicateWithParams(between, '{0} is not between {1} and {2}')(1, 9)
    matches_matches = [2, 4, 6, 8]
    matches_mismatches = [0, 1, 9, 10]
    str_examples = [('MatchesPredicateWithParams({!r}, {!r})({})'.format(between, '{0} is not between {1} and {2}', '1, 2'), MatchesPredicateWithParams(between, '{0} is not between {1} and {2}')(1, 2)), ('Between(1, 2)', MatchesPredicateWithParams(between, '{0} is not between {1} and {2}', 'Between')(1, 2))]
    describe_examples = [('1 is not between 2 and 3', 1, MatchesPredicateWithParams(between, '{0} is not between {1} and {2}')(2, 3))]