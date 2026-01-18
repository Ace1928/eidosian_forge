import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class DoesNotStartWithTests(TestCase):
    run_tests_with = FullStackRunTest

    def test_describe(self):
        mismatch = DoesNotStartWith('fo', 'bo')
        self.assertEqual("'fo' does not start with 'bo'.", mismatch.describe())

    def test_describe_non_ascii_unicode(self):
        string = 'A§'
        suffix = 'B§'
        mismatch = DoesNotStartWith(string, suffix)
        self.assertEqual('{} does not start with {}.'.format(text_repr(string), text_repr(suffix)), mismatch.describe())

    def test_describe_non_ascii_bytes(self):
        string = _b('A§')
        suffix = _b('B§')
        mismatch = DoesNotStartWith(string, suffix)
        self.assertEqual(f'{string!r} does not start with {suffix!r}.', mismatch.describe())