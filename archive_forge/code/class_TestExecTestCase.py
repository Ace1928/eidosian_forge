import datetime
import io
import os
import tempfile
import unittest
from io import BytesIO
from testtools import PlaceHolder, TestCase, TestResult, skipIf
from testtools.compat import _b, _u
from testtools.content import Content, TracebackContent, text_content
from testtools.content_type import ContentType
from testtools.matchers import Contains, Equals, MatchesAny
import iso8601
import subunit
from subunit.tests import (_remote_exception_repr,
class TestExecTestCase(unittest.TestCase):

    class SampleExecTestCase(subunit.ExecTestCase):

        def test_sample_method(self):
            """sample-script.py"""

        def test_sample_method_args(self):
            """sample-script.py foo"""

    def test_construct(self):
        test = self.SampleExecTestCase('test_sample_method')
        self.assertEqual(test.script, subunit.join_dir(__file__, 'sample-script.py'))

    def test_args(self):
        result = unittest.TestResult()
        test = self.SampleExecTestCase('test_sample_method_args')
        test.run(result)
        self.assertEqual(1, result.testsRun)

    def test_run(self):
        result = ExtendedTestResult()
        test = self.SampleExecTestCase('test_sample_method')
        test.run(result)
        mcdonald = subunit.RemotedTestCase('old mcdonald')
        bing = subunit.RemotedTestCase('bing crosby')
        bing_details = {}
        bing_details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b('foo.c:53:ERROR invalid state\n')])
        an_error = subunit.RemotedTestCase('an error')
        error_details = {}
        self.assertEqual([('startTest', mcdonald), ('addSuccess', mcdonald), ('stopTest', mcdonald), ('startTest', bing), ('addFailure', bing, bing_details), ('stopTest', bing), ('startTest', an_error), ('addError', an_error, error_details), ('stopTest', an_error)], result._events)

    def test_debug(self):
        test = self.SampleExecTestCase('test_sample_method')
        test.debug()

    def test_count_test_cases(self):
        """TODO run the child process and count responses to determine the count."""

    def test_join_dir(self):
        sibling = subunit.join_dir(__file__, 'foo')
        filedir = os.path.abspath(os.path.dirname(__file__))
        expected = os.path.join(filedir, 'foo')
        self.assertEqual(sibling, expected)