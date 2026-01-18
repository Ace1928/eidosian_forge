from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
class InterruptInTestTests(TrialTest):
    test_03_doNothing_run: bool | None

    class InterruptedTest(unittest.TestCase):

        def test_02_raiseInterrupt(self) -> None:
            raise KeyboardInterrupt

        def test_01_doNothing(self) -> None:
            pass

        def test_03_doNothing(self) -> None:
            InterruptInTestTests.test_03_doNothing_run = True

    def setUp(self) -> None:
        super().setUp()
        self.suite = self.loader.loadClass(InterruptInTestTests.InterruptedTest)
        InterruptInTestTests.test_03_doNothing_run = None

    def test_setUpOK(self) -> None:
        self.assertEqual(3, self.suite.countTestCases())
        self.assertEqual(0, self.reporter.testsRun)
        self.assertFalse(self.reporter.shouldStop)

    def test_interruptInTest(self) -> None:
        runner.TrialSuite([self.suite]).run(self.reporter)
        self.assertTrue(self.reporter.shouldStop)
        self.assertEqual(2, self.reporter.testsRun)
        self.assertFalse(InterruptInTestTests.test_03_doNothing_run, 'test_03_doNothing ran.')