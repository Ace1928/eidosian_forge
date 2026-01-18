import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestRaisesConvenience(TestCase):
    run_tests_with = FullStackRunTest

    def test_exc_type(self):
        self.assertThat(lambda: 1 / 0, raises(ZeroDivisionError))

    def test_exc_value(self):
        e = RuntimeError('You lose!')

        def raiser():
            raise e
        self.assertThat(raiser, raises(e))