from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchersAnyInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesAny(DocTestMatches('1'), DocTestMatches('2'))
    matches_matches = ['1', '2']
    matches_mismatches = ['3']
    str_examples = [("MatchesAny(DocTestMatches('1\\n'), DocTestMatches('2\\n'))", MatchesAny(DocTestMatches('1'), DocTestMatches('2')))]
    describe_examples = [('Differences: [\nExpected:\n    1\nGot:\n    3\n\nExpected:\n    2\nGot:\n    3\n\n]', '3', MatchesAny(DocTestMatches('1'), DocTestMatches('2')))]