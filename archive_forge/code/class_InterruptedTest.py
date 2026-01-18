from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
class InterruptedTest(unittest.TestCase):

    def tearDown(self) -> None:
        if InterruptInTearDownTests.testsRun > 0:
            raise KeyboardInterrupt

    def test_01(self) -> None:
        InterruptInTearDownTests.testsRun += 1

    def test_02(self) -> None:
        InterruptInTearDownTests.testsRun += 1
        InterruptInTearDownTests.test_02_run = True