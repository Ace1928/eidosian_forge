import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
class FullStackRunTest(runtest.RunTest):

    def _run_user(self, fn, *args, **kwargs):
        return run_with_stack_hidden(False, super()._run_user, fn, *args, **kwargs)