from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesPredicate(TestCase, TestMatchersInterface):
    matches_matcher = MatchesPredicate(is_even, '%s is not even')
    matches_matches = [2, 4, 6, 8]
    matches_mismatches = [3, 5, 7, 9]
    str_examples = [('MatchesPredicate({!r}, {!r})'.format(is_even, '%s is not even'), MatchesPredicate(is_even, '%s is not even'))]
    describe_examples = [('7 is not even', 7, MatchesPredicate(is_even, '%s is not even'))]