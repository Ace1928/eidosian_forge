import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class ErrorInTearDown(Base):
    expected_calls = ['setUp', 'test', 'clean-up']
    expected_results = [('addError', RuntimeError)]

    def tearDown(self):
        raise RuntimeError('Error in tearDown')