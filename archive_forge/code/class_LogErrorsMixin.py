import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
class LogErrorsMixin:
    """
    High-level tests demonstrating the expected behaviour of logged errors
    during tests.
    """

    def setUp(self):
        self.result = reporter.TestResult()

    def tearDown(self):
        self.flushLoggedErrors(ZeroDivisionError)

    def test_singleError(self):
        """
        Test that a logged error gets reported as a test error.
        """
        test = self.MockTest('test_single')
        test(self.result)
        self.assertEqual(len(self.result.errors), 1)
        self.assertTrue(self.result.errors[0][1].check(ZeroDivisionError), self.result.errors[0][1])
        self.assertEqual(0, self.result.successes)

    def test_twoErrors(self):
        """
        Test that when two errors get logged, they both get reported as test
        errors.
        """
        test = self.MockTest('test_double')
        test(self.result)
        self.assertEqual(len(self.result.errors), 2)
        self.assertEqual(0, self.result.successes)

    def test_errorsIsolated(self):
        """
        Check that an error logged in one test doesn't fail the next test.
        """
        t1 = self.MockTest('test_single')
        t2 = self.MockTest('test_silent')
        t1(self.result)
        t2(self.result)
        self.assertEqual(len(self.result.errors), 1)
        self.assertEqual(self.result.errors[0][0], t1)
        self.assertEqual(1, self.result.successes)

    def test_errorsIsolatedWhenTestFails(self):
        """
        An error logged in a failed test doesn't fail the next test.
        """
        t1 = self.MockTest('test_singleThenFail')
        t2 = self.MockTest('test_silent')
        t1(self.result)
        t2(self.result)
        self.assertEqual(len(self.result.errors), 2)
        self.assertEqual(self.result.errors[0][0], t1)
        self.result.errors[0][1].trap(TypeError)
        self.assertEqual(self.result.errors[1][0], t1)
        self.result.errors[1][1].trap(ZeroDivisionError)
        self.assertEqual(1, self.result.successes)

    def test_boundedObservers(self):
        """
        There are no extra log observers after a test runs.
        """
        observer = _synctest._LogObserver()
        self.patch(_synctest, '_logObserver', observer)
        observers = log.theLogPublisher.observers[:]
        test = self.MockTest()
        test(self.result)
        self.assertEqual(observers, log.theLogPublisher.observers)