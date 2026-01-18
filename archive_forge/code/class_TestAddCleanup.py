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
class TestAddCleanup(TestCase):
    """Tests for TestCase.addCleanup."""
    run_tests_with = FullStackRunTest

    def test_cleanup_run_after_tearDown(self):
        log = []
        test = make_test_case(self.getUniqueString(), set_up=lambda _: log.append('setUp'), test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('cleanup')])
        test.run()
        self.assertThat(log, Equals(['setUp', 'runTest', 'tearDown', 'cleanup']))

    def test_add_cleanup_called_if_setUp_fails(self):
        log = []

        def broken_set_up(ignored):
            log.append('brokenSetUp')
            raise RuntimeError('Deliberate broken setUp')
        test = make_test_case(self.getUniqueString(), set_up=broken_set_up, test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('cleanup')])
        test.run()
        self.assertThat(log, Equals(['brokenSetUp', 'cleanup']))

    def test_addCleanup_called_in_reverse_order(self):
        log = []
        test = make_test_case(self.getUniqueString(), set_up=lambda _: log.append('setUp'), test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('first'), lambda _: log.append('second')])
        test.run()
        self.assertThat(log, Equals(['setUp', 'runTest', 'tearDown', 'second', 'first']))

    def test_tearDown_runs_on_cleanup_failure(self):
        log = []
        test = make_test_case(self.getUniqueString(), set_up=lambda _: log.append('setUp'), test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: 1 / 0])
        test.run()
        self.assertThat(log, Equals(['setUp', 'runTest', 'tearDown']))

    def test_cleanups_continue_running_after_error(self):
        log = []
        test = make_test_case(self.getUniqueString(), set_up=lambda _: log.append('setUp'), test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('first'), lambda _: 1 / 0, lambda _: log.append('second')])
        test.run()
        self.assertThat(log, Equals(['setUp', 'runTest', 'tearDown', 'second', 'first']))

    def test_error_in_cleanups_are_captured(self):
        test = make_test_case(self.getUniqueString(), cleanups=[lambda _: 1 / 0])
        log = []
        test.run(ExtendedTestResult(log))
        self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))

    def test_keyboard_interrupt_not_caught(self):
        test = make_test_case(self.getUniqueString(), cleanups=[lambda _: raise_(KeyboardInterrupt())])
        self.assertThat(test.run, Raises(MatchesException(KeyboardInterrupt)))

    def test_all_errors_from_MultipleExceptions_reported(self):

        def raise_many(ignored):
            try:
                1 / 0
            except Exception:
                exc_info1 = sys.exc_info()
            try:
                1 / 0
            except Exception:
                exc_info2 = sys.exc_info()
            raise MultipleExceptions(exc_info1, exc_info2)
        test = make_test_case(self.getUniqueString(), cleanups=[raise_many])
        log = []
        test.run(ExtendedTestResult(log))
        self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError'])), 'traceback-1': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))

    def test_multipleCleanupErrorsReported(self):
        test = make_test_case(self.getUniqueString(), cleanups=[lambda _: 1 / 0, lambda _: 1 / 0])
        log = []
        test.run(ExtendedTestResult(log))
        self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError'])), 'traceback-1': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))

    def test_multipleErrorsCoreAndCleanupReported(self):
        test = make_test_case(self.getUniqueString(), test_body=lambda _: raise_(RuntimeError('Deliberately broken test')), cleanups=[lambda _: 1 / 0, lambda _: 1 / 0])
        log = []
        test.run(ExtendedTestResult(log))
        self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'RuntimeError: Deliberately broken test'])), 'traceback-1': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError'])), 'traceback-2': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))