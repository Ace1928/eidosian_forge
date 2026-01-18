from testtools import TestCase
from testtools.matchers import Always, Never
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestAlwaysInterface(TestMatchersInterface, TestCase):
    """:py:func:`~testtools.matchers.Always` always matches."""
    matches_matcher = Always()
    matches_matches = [42, object(), 'hi mom']
    matches_mismatches = []
    str_examples = [('Always()', Always())]
    describe_examples = []