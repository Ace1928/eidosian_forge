import doctest
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._doctest import DocTestMatches
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestDocTestMatchesInterface(TestCase, TestMatchersInterface):
    matches_matcher = DocTestMatches('Ran 1 test in ...s', doctest.ELLIPSIS)
    matches_matches = ['Ran 1 test in 0.000s', 'Ran 1 test in 1.234s']
    matches_mismatches = ['Ran 1 tests in 0.000s', 'Ran 2 test in 0.000s']
    str_examples = [("DocTestMatches('Ran 1 test in ...s\\n')", DocTestMatches('Ran 1 test in ...s')), ("DocTestMatches('foo\\n', flags=8)", DocTestMatches('foo', flags=8))]
    describe_examples = [('Expected:\n    Ran 1 tests in ...s\nGot:\n    Ran 1 test in 0.123s\n', 'Ran 1 test in 0.123s', DocTestMatches('Ran 1 tests in ...s', doctest.ELLIPSIS))]