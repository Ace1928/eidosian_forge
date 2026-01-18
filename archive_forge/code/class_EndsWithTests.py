import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class EndsWithTests(TestCase):
    run_tests_with = FullStackRunTest

    def test_str(self):
        matcher = EndsWith('bar')
        self.assertEqual("EndsWith('bar')", str(matcher))

    def test_str_with_bytes(self):
        b = _b('§')
        matcher = EndsWith(b)
        self.assertEqual(f'EndsWith({b!r})', str(matcher))

    def test_str_with_unicode(self):
        u = '§'
        matcher = EndsWith(u)
        self.assertEqual(f'EndsWith({u!r})', str(matcher))

    def test_match(self):
        matcher = EndsWith('arf')
        self.assertIs(None, matcher.match('barf'))

    def test_mismatch_returns_does_not_end_with(self):
        matcher = EndsWith('bar')
        self.assertIsInstance(matcher.match('foo'), DoesNotEndWith)

    def test_mismatch_sets_matchee(self):
        matcher = EndsWith('bar')
        mismatch = matcher.match('foo')
        self.assertEqual('foo', mismatch.matchee)

    def test_mismatch_sets_expected(self):
        matcher = EndsWith('bar')
        mismatch = matcher.match('foo')
        self.assertEqual('bar', mismatch.expected)