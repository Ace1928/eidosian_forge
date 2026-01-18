from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
class CustomRunTest(RunTest):
    marker = object()

    def run(self, result=None):
        return self.marker