from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
class FooRunTest(RunTest):

    def __init__(self, case, handlers=None, bar=None):
        super().__init__(case, handlers)
        self.bar = bar

    def run(self, result=None):
        return self.bar