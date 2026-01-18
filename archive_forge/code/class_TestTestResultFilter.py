import subprocess
import sys
import unittest
from datetime import datetime
from io import BytesIO
from testtools import TestCase
from testtools.compat import _b
from testtools.testresult.doubles import ExtendedTestResult, StreamResult
import iso8601
import subunit
from subunit.test_results import make_tag_filter, TestResultFilter
from subunit import ByteStreamToStreamResult, StreamResultToBytes
class TestTestResultFilter(TestCase):
    """Test for TestResultFilter, a TestResult object which filters tests."""
    example_subunit_stream = _b('tags: global\ntest passed\nsuccess passed\ntest failed\ntags: local\nfailure failed\ntest error\nerror error [\nerror details\n]\ntest skipped\nskip skipped\ntest todo\nxfail todo\n')

    def run_tests(self, result_filter, input_stream=None):
        """Run tests through the given filter.

        :param result_filter: A filtering TestResult object.
        :param input_stream: Bytes of subunit stream data. If not provided,
            uses TestTestResultFilter.example_subunit_stream.
        """
        if input_stream is None:
            input_stream = self.example_subunit_stream
        test = subunit.ProtocolTestCase(BytesIO(input_stream))
        test.run(result_filter)

    def test_default(self):
        """The default is to exclude success and include everything else."""
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result)
        self.run_tests(result_filter)
        self.assertEqual(['error'], [error[0].id() for error in filtered_result.errors])
        self.assertEqual(['failed'], [failure[0].id() for failure in filtered_result.failures])
        self.assertEqual(4, filtered_result.testsRun)

    def test_tag_filter(self):
        tag_filter = make_tag_filter(['global'], ['local'])
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result, filter_success=False, filter_predicate=tag_filter)
        self.run_tests(result_filter)
        tests_included = [event[1] for event in result._events if event[0] == 'startTest']
        tests_expected = list(map(subunit.RemotedTestCase, ['passed', 'error', 'skipped', 'todo']))
        self.assertEqual(tests_expected, tests_included)

    def test_tags_tracked_correctly(self):
        tag_filter = make_tag_filter(['a'], [])
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result, filter_success=False, filter_predicate=tag_filter)
        input_stream = _b('test: foo\ntags: a\nsuccessful: foo\ntest: bar\nsuccessful: bar\n')
        self.run_tests(result_filter, input_stream)
        foo = subunit.RemotedTestCase('foo')
        self.assertEqual([('startTest', foo), ('tags', {'a'}, set()), ('addSuccess', foo), ('stopTest', foo)], result._events)

    def test_exclude_errors(self):
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, filter_error=True)
        self.run_tests(result_filter)
        self.assertEqual([], filtered_result.errors)
        self.assertEqual(['failed'], [failure[0].id() for failure in filtered_result.failures])
        self.assertEqual(3, filtered_result.testsRun)

    def test_fixup_expected_failures(self):
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, fixup_expected_failures={'failed'})
        self.run_tests(result_filter)
        self.assertEqual(['failed', 'todo'], [failure[0].id() for failure in filtered_result.expectedFailures])
        self.assertEqual([], filtered_result.failures)
        self.assertEqual(4, filtered_result.testsRun)

    def test_fixup_expected_errors(self):
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, fixup_expected_failures={'error'})
        self.run_tests(result_filter)
        self.assertEqual(['error', 'todo'], [failure[0].id() for failure in filtered_result.expectedFailures])
        self.assertEqual([], filtered_result.errors)
        self.assertEqual(4, filtered_result.testsRun)

    def test_fixup_unexpected_success(self):
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, filter_success=False, fixup_expected_failures={'passed'})
        self.run_tests(result_filter)
        self.assertEqual(['passed'], [passed.id() for passed in filtered_result.unexpectedSuccesses])
        self.assertEqual(5, filtered_result.testsRun)

    def test_exclude_failure(self):
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, filter_failure=True)
        self.run_tests(result_filter)
        self.assertEqual(['error'], [error[0].id() for error in filtered_result.errors])
        self.assertEqual([], [failure[0].id() for failure in filtered_result.failures])
        self.assertEqual(3, filtered_result.testsRun)

    def test_exclude_skips(self):
        filtered_result = subunit.TestResultStats(None)
        result_filter = TestResultFilter(filtered_result, filter_skip=True)
        self.run_tests(result_filter)
        self.assertEqual(0, filtered_result.skipped_tests)
        self.assertEqual(2, filtered_result.failed_tests)
        self.assertEqual(3, filtered_result.testsRun)

    def test_include_success(self):
        """Successes can be included if requested."""
        filtered_result = unittest.TestResult()
        result_filter = TestResultFilter(filtered_result, filter_success=False)
        self.run_tests(result_filter)
        self.assertEqual(['error'], [error[0].id() for error in filtered_result.errors])
        self.assertEqual(['failed'], [failure[0].id() for failure in filtered_result.failures])
        self.assertEqual(5, filtered_result.testsRun)

    def test_filter_predicate(self):
        """You can filter by predicate callbacks"""
        filtered_result = unittest.TestResult()

        def filter_cb(test, outcome, err, details):
            return outcome == 'success'
        result_filter = TestResultFilter(filtered_result, filter_predicate=filter_cb, filter_success=False)
        self.run_tests(result_filter)
        self.assertEqual(1, filtered_result.testsRun)

    def test_filter_predicate_with_tags(self):
        """You can filter by predicate callbacks that accept tags"""
        filtered_result = unittest.TestResult()

        def filter_cb(test, outcome, err, details, tags):
            return outcome == 'success'
        result_filter = TestResultFilter(filtered_result, filter_predicate=filter_cb, filter_success=False)
        self.run_tests(result_filter)
        self.assertEqual(1, filtered_result.testsRun)

    def test_time_ordering_preserved(self):
        date_a = datetime(year=2000, month=1, day=1, tzinfo=iso8601.UTC)
        date_b = datetime(year=2000, month=1, day=2, tzinfo=iso8601.UTC)
        date_c = datetime(year=2000, month=1, day=3, tzinfo=iso8601.UTC)
        subunit_stream = _b('\n'.join(['time: %s', 'test: foo', 'time: %s', 'error: foo', 'time: %s', '']) % (date_a, date_b, date_c))
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result)
        self.run_tests(result_filter, subunit_stream)
        foo = subunit.RemotedTestCase('foo')
        self.maxDiff = None
        self.assertEqual([('time', date_a), ('time', date_b), ('startTest', foo), ('addError', foo, {}), ('stopTest', foo), ('time', date_c)], result._events)

    def test_time_passes_through_filtered_tests(self):
        date_a = datetime(year=2000, month=1, day=1, tzinfo=iso8601.UTC)
        date_b = datetime(year=2000, month=1, day=2, tzinfo=iso8601.UTC)
        date_c = datetime(year=2000, month=1, day=3, tzinfo=iso8601.UTC)
        subunit_stream = _b('\n'.join(['time: %s', 'test: foo', 'time: %s', 'success: foo', 'time: %s', '']) % (date_a, date_b, date_c))
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result)
        result_filter.startTestRun()
        self.run_tests(result_filter, subunit_stream)
        result_filter.stopTestRun()
        subunit.RemotedTestCase('foo')
        self.maxDiff = None
        self.assertEqual([('startTestRun',), ('time', date_a), ('time', date_c), ('stopTestRun',)], result._events)

    def test_skip_preserved(self):
        subunit_stream = _b('\n'.join(['test: foo', 'skip: foo', '']))
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result)
        self.run_tests(result_filter, subunit_stream)
        foo = subunit.RemotedTestCase('foo')
        self.assertEqual([('startTest', foo), ('addSkip', foo, {}), ('stopTest', foo)], result._events)

    def test_renames(self):

        def rename(name):
            return name + ' - renamed'
        result = ExtendedTestResult()
        result_filter = TestResultFilter(result, filter_success=False, rename=rename)
        input_stream = _b('test: foo\nsuccessful: foo\n')
        self.run_tests(result_filter, input_stream)
        self.assertEqual([('startTest', 'foo - renamed'), ('addSuccess', 'foo - renamed'), ('stopTest', 'foo - renamed')], [(ev[0], ev[1].id()) for ev in result._events])