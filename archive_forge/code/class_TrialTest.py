from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
class TrialTest(unittest.SynchronousTestCase):

    def setUp(self) -> None:
        self.output = StringIO()
        self.reporter = reporter.TestResult()
        self.loader = runner.TestLoader()