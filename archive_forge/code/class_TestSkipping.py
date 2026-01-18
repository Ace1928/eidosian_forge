from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestSkipping(TestCase):
    """Tests for skipping of tests functionality."""
    run_tests_with = FullStackRunTest

    def test_skip_causes_skipException(self):
        self.assertThat(lambda: self.skipTest('Skip this test'), Raises(MatchesException(self.skipException)))

    def test_can_use_skipTest(self):
        self.assertThat(lambda: self.skipTest('Skip this test'), Raises(MatchesException(self.skipException)))

    def test_skip_without_reason_works(self):

        class Test(TestCase):

            def test(self):
                raise self.skipException()
        case = Test('test')
        result = ExtendedTestResult()
        case.run(result)
        self.assertEqual('addSkip', result._events[1][0])
        self.assertEqual('no reason given.', result._events[1][2]['reason'].as_text())

    def test_skipException_in_setup_calls_result_addSkip(self):

        class TestThatRaisesInSetUp(TestCase):

            def setUp(self):
                TestCase.setUp(self)
                self.skipTest('skipping this test')

            def test_that_passes(self):
                pass
        calls = []
        result = LoggingResult(calls)
        test = TestThatRaisesInSetUp('test_that_passes')
        test.run(result)
        case = result._events[0][1]
        self.assertEqual([('startTest', case), ('addSkip', case, 'skipping this test'), ('stopTest', case)], calls)

    def test_skipException_in_test_method_calls_result_addSkip(self):

        class SkippingTest(TestCase):

            def test_that_raises_skipException(self):
                self.skipTest('skipping this test')
        events = []
        result = Python27TestResult(events)
        test = SkippingTest('test_that_raises_skipException')
        test.run(result)
        case = result._events[0][1]
        self.assertEqual([('startTest', case), ('addSkip', case, 'skipping this test'), ('stopTest', case)], events)

    def test_different_skipException_in_test_method_calls_result_addSkip(self):

        class SkippingTest(TestCase):
            skipException = ValueError

            def test_that_raises_skipException(self):
                self.skipTest('skipping this test')
        events = []
        result = Python27TestResult(events)
        test = SkippingTest('test_that_raises_skipException')
        test.run(result)
        case = result._events[0][1]
        self.assertEqual([('startTest', case), ('addSkip', case, 'skipping this test'), ('stopTest', case)], events)

    def test_skip__in_setup_with_old_result_object_calls_addSuccess(self):

        class SkippingTest(TestCase):

            def setUp(self):
                TestCase.setUp(self)
                raise self.skipException('skipping this test')

            def test_that_raises_skipException(self):
                pass
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_that_raises_skipException')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skip_with_old_result_object_calls_addError(self):

        class SkippingTest(TestCase):

            def test_that_raises_skipException(self):
                raise self.skipException('skipping this test')
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_that_raises_skipException')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skip_decorator(self):

        class SkippingTest(TestCase):

            @skip('skipping this test')
            def test_that_is_decorated_with_skip(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_that_is_decorated_with_skip')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skipIf_decorator(self):

        class SkippingTest(TestCase):

            @skipIf(True, 'skipping this test')
            def test_that_is_decorated_with_skipIf(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_that_is_decorated_with_skipIf')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skipUnless_decorator(self):

        class SkippingTest(TestCase):

            @skipUnless(False, 'skipping this test')
            def test_that_is_decorated_with_skipUnless(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_that_is_decorated_with_skipUnless')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skip_decorator_shared(self):

        def shared(testcase):
            testcase.fail('nope')

        class SkippingTest(TestCase):
            test_skip = skipIf(True, 'skipping this test')(shared)

        class NotSkippingTest(TestCase):
            test_no_skip = skipIf(False, 'skipping this test')(shared)
        events = []
        result = Python26TestResult(events)
        test = SkippingTest('test_skip')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])
        events2 = []
        result2 = Python26TestResult(events2)
        test2 = NotSkippingTest('test_no_skip')
        test2.run(result2)
        self.assertEqual('addFailure', events2[1][0])

    def test_skip_class_decorator(self):

        @skip('skipping this testcase')
        class SkippingTest(TestCase):

            def test_that_is_decorated_with_skip(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        try:
            test = SkippingTest('test_that_is_decorated_with_skip')
        except TestSkipped:
            self.fail('TestSkipped raised')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skipIf_class_decorator(self):

        @skipIf(True, 'skipping this testcase')
        class SkippingTest(TestCase):

            def test_that_is_decorated_with_skipIf(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        try:
            test = SkippingTest('test_that_is_decorated_with_skipIf')
        except TestSkipped:
            self.fail('TestSkipped raised')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def test_skipUnless_class_decorator(self):

        @skipUnless(False, 'skipping this testcase')
        class SkippingTest(TestCase):

            def test_that_is_decorated_with_skipUnless(self):
                self.fail()
        events = []
        result = Python26TestResult(events)
        try:
            test = SkippingTest('test_that_is_decorated_with_skipUnless')
        except TestSkipped:
            self.fail('TestSkipped raised')
        test.run(result)
        self.assertEqual('addSuccess', events[1][0])

    def check_skip_decorator_does_not_run_setup(self, decorator, reason):

        class SkippingTest(TestCase):
            setup_ran = False

            def setUp(self):
                super().setUp()
                self.setup_ran = True

            @decorator
            def test_skipped(self):
                self.fail()
        test = SkippingTest('test_skipped')
        self.check_test_does_not_run_setup(test, reason)

        @decorator
        class SkippingTestCase(TestCase):
            setup_ran = False

            def setUp(self):
                super().setUp()
                self.setup_ran = True

            def test_skipped(self):
                self.fail()
        try:
            test = SkippingTestCase('test_skipped')
        except TestSkipped:
            self.fail('TestSkipped raised')
        self.check_test_does_not_run_setup(test, reason)

    def check_test_does_not_run_setup(self, test, reason):
        result = test.run()
        self.assertTrue(result.wasSuccessful())
        self.assertIn(reason, result.skip_reasons, result.skip_reasons)
        self.assertFalse(test.setup_ran)

    def test_testtools_skip_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(skip(reason), reason)

    def test_testtools_skipIf_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(skipIf(True, reason), reason)

    def test_testtools_skipUnless_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(skipUnless(False, reason), reason)

    def test_unittest_skip_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(unittest.skip(reason), reason)

    def test_unittest_skipIf_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(unittest.skipIf(True, reason), reason)

    def test_unittest_skipUnless_decorator_does_not_run_setUp(self):
        reason = self.getUniqueString()
        self.check_skip_decorator_does_not_run_setup(unittest.skipUnless(False, reason), reason)