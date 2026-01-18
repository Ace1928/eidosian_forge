import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
class _TextAndXMLTestResult(_pretty_print_reporter.TextTestResult):
    """Private TestResult class that produces both formatted text results and XML.

  Used by TextAndXMLTestRunner.
  """
    _TEST_SUITE_RESULT_CLASS = _TestSuiteResult
    _TEST_CASE_RESULT_CLASS = _TestCaseResult

    def __init__(self, xml_stream, stream, descriptions, verbosity, time_getter=_time_copy, testsuites_properties=None):
        super(_TextAndXMLTestResult, self).__init__(stream, descriptions, verbosity)
        self.xml_stream = xml_stream
        self.pending_test_case_results = {}
        self.suite = self._TEST_SUITE_RESULT_CLASS()
        if testsuites_properties:
            self.suite._testsuites_properties = testsuites_properties
        self.time_getter = time_getter
        self._pending_test_case_results_lock = threading.RLock()

    def startTest(self, test):
        self.start_time = self.time_getter()
        super(_TextAndXMLTestResult, self).startTest(test)

    def stopTest(self, test):
        with self._pending_test_case_results_lock:
            super(_TextAndXMLTestResult, self).stopTest(test)
            result = self.get_pending_test_case_result(test)
            if not result:
                test_name = test.id() or str(test)
                sys.stderr.write('No pending test case: %s\n' % test_name)
                return
            if getattr(self, 'start_time', None) is None:
                self.start_time = self.time_getter()
            test_id = id(test)
            run_time = self.time_getter() - self.start_time
            result.set_run_time(run_time)
            result.set_start_time(self.start_time)
            self.suite.add_test_case_result(result)
            del self.pending_test_case_results[test_id]

    def startTestRun(self):
        self.suite.set_start_time(self.time_getter())
        super(_TextAndXMLTestResult, self).startTestRun()

    def stopTestRun(self):
        self.suite.set_end_time(self.time_getter())
        with self._pending_test_case_results_lock:
            for test_id in self.pending_test_case_results:
                result = self.pending_test_case_results[test_id]
                if getattr(self, 'start_time', None) is not None:
                    run_time = self.suite.overall_end_time - self.start_time
                    result.set_run_time(run_time)
                    result.set_start_time(self.start_time)
                self.suite.add_test_case_result(result)
            self.pending_test_case_results.clear()

    def _exc_info_to_string(self, err, test=None):
        """Converts a sys.exc_info()-style tuple of values into a string.

    This method must be overridden because the method signature in
    unittest.TestResult changed between Python 2.2 and 2.4.

    Args:
      err: A sys.exc_info() tuple of values for an error.
      test: The test method.

    Returns:
      A formatted exception string.
    """
        if test:
            return super(_TextAndXMLTestResult, self)._exc_info_to_string(err, test)
        return ''.join(traceback.format_exception(*err))

    def add_pending_test_case_result(self, test, error_summary=None, skip_reason=None):
        """Adds result information to a test case result which may still be running.

    If a result entry for the test already exists, add_pending_test_case_result
    will add error summary tuples and/or overwrite skip_reason for the result.
    If it does not yet exist, a result entry will be created.
    Note that a test result is considered to have been run and passed
    only if there are no errors or skip_reason.

    Args:
      test: A test method as defined by unittest
      error_summary: A 4-tuple with the following entries:
          1) a string identifier of either "failure" or "error"
          2) an exception_type
          3) an exception_message
          4) a string version of a sys.exc_info()-style tuple of values
             ('error', err[0], err[1], self._exc_info_to_string(err))
             If the length of errors is 0, then the test is either passed or
             skipped.
      skip_reason: a string explaining why the test was skipped
    """
        with self._pending_test_case_results_lock:
            test_id = id(test)
            if test_id not in self.pending_test_case_results:
                self.pending_test_case_results[test_id] = self._TEST_CASE_RESULT_CLASS(test)
            if error_summary:
                self.pending_test_case_results[test_id].errors.append(error_summary)
            if skip_reason:
                self.pending_test_case_results[test_id].skip_reason = skip_reason

    def delete_pending_test_case_result(self, test):
        with self._pending_test_case_results_lock:
            test_id = id(test)
            del self.pending_test_case_results[test_id]

    def get_pending_test_case_result(self, test):
        test_id = id(test)
        return self.pending_test_case_results.get(test_id, None)

    def addSuccess(self, test):
        super(_TextAndXMLTestResult, self).addSuccess(test)
        self.add_pending_test_case_result(test)

    def addError(self, test, err):
        super(_TextAndXMLTestResult, self).addError(test, err)
        error_summary = ('error', err[0], err[1], self._exc_info_to_string(err, test=test))
        self.add_pending_test_case_result(test, error_summary=error_summary)

    def addFailure(self, test, err):
        super(_TextAndXMLTestResult, self).addFailure(test, err)
        error_summary = ('failure', err[0], err[1], self._exc_info_to_string(err, test=test))
        self.add_pending_test_case_result(test, error_summary=error_summary)

    def addSkip(self, test, reason):
        super(_TextAndXMLTestResult, self).addSkip(test, reason)
        self.add_pending_test_case_result(test, skip_reason=reason)

    def addExpectedFailure(self, test, err):
        super(_TextAndXMLTestResult, self).addExpectedFailure(test, err)
        if callable(getattr(test, 'recordProperty', None)):
            test.recordProperty('EXPECTED_FAILURE', self._exc_info_to_string(err, test=test))
        self.add_pending_test_case_result(test)

    def addUnexpectedSuccess(self, test):
        super(_TextAndXMLTestResult, self).addUnexpectedSuccess(test)
        test_name = test.id() or str(test)
        error_summary = ('error', '', '', 'Test case %s should have failed, but passed.' % test_name)
        self.add_pending_test_case_result(test, error_summary=error_summary)

    def addSubTest(self, test, subtest, err):
        super(_TextAndXMLTestResult, self).addSubTest(test, subtest, err)
        if err is not None:
            if issubclass(err[0], test.failureException):
                error_summary = ('failure', err[0], err[1], self._exc_info_to_string(err, test=test))
            else:
                error_summary = ('error', err[0], err[1], self._exc_info_to_string(err, test=test))
        else:
            error_summary = None
        self.add_pending_test_case_result(subtest, error_summary=error_summary)

    def printErrors(self):
        super(_TextAndXMLTestResult, self).printErrors()
        self.xml_stream.write('<?xml version="1.0"?>\n')
        self.suite.print_xml_summary(self.xml_stream)