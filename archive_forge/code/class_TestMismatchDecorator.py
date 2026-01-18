from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
class TestMismatchDecorator(TestCase):
    run_tests_with = FullStackRunTest

    def test_forwards_description(self):
        x = Mismatch('description', {'foo': 'bar'})
        decorated = MismatchDecorator(x)
        self.assertEqual(x.describe(), decorated.describe())

    def test_forwards_details(self):
        x = Mismatch('description', {'foo': 'bar'})
        decorated = MismatchDecorator(x)
        self.assertEqual(x.get_details(), decorated.get_details())

    def test_repr(self):
        x = Mismatch('description', {'foo': 'bar'})
        decorated = MismatchDecorator(x)
        self.assertEqual(f'<testtools.matchers.MismatchDecorator({x!r})>', repr(decorated))