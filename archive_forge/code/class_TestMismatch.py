from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
class TestMismatch(TestCase):
    run_tests_with = FullStackRunTest

    def test_constructor_arguments(self):
        mismatch = Mismatch('some description', {'detail': 'things'})
        self.assertEqual('some description', mismatch.describe())
        self.assertEqual({'detail': 'things'}, mismatch.get_details())

    def test_constructor_no_arguments(self):
        mismatch = Mismatch()
        self.assertThat(mismatch.describe, Raises(MatchesException(NotImplementedError)))
        self.assertEqual({}, mismatch.get_details())