import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class ExpectThatFailure(Base):
    """Calling expectThat with a failing match fails the test."""
    expected_calls = ['setUp', 'test', 'tearDown', 'clean-up']
    expected_results = [('addFailure', AssertionError)]

    def test_something(self):
        self.calls.append('test')
        self.expectThat(object(), Is(object()))