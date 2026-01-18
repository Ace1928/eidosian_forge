import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class ErrorInCleanup(Base):
    expected_calls = ['setUp', 'test', 'tearDown', 'clean-up']
    expected_results = [('addError', ZeroDivisionError)]

    def test_something(self):
        self.calls.append('test')
        self.addCleanup(lambda: 1 / 0)