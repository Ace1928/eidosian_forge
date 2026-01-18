import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesExceptionTypeMatcherInterface(TestCase, TestMatchersInterface):
    matches_matcher = MatchesException(ValueError, AfterPreprocessing(str, Equals('foo')))
    error_foo = make_error(ValueError, 'foo')
    error_sub = make_error(UnicodeError, 'foo')
    error_bar = make_error(ValueError, 'bar')
    matches_matches = [error_foo, error_sub]
    matches_mismatches = [error_bar]
    str_examples = [('MatchesException(%r)' % Exception, MatchesException(Exception, Equals('foo')))]
    describe_examples = [(f'{error_bar[1]!r} != 5', error_bar, MatchesException(ValueError, Equals(5)))]