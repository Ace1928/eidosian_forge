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
class TestIsolatedTestSuite(TestCase):

    class SampleTestToIsolate(unittest.TestCase):
        SETUP = False
        TEARDOWN = False
        TEST = False

        def setUp(self):
            TestIsolatedTestSuite.SampleTestToIsolate.SETUP = True

        def tearDown(self):
            TestIsolatedTestSuite.SampleTestToIsolate.TEARDOWN = True

        def test_sets_global_state(self):
            TestIsolatedTestSuite.SampleTestToIsolate.TEST = True

    def test_construct(self):
        subunit.IsolatedTestSuite()

    @skipIf(os.name != 'posix', 'Need a posix system for forking tests')
    def test_run(self):
        result = unittest.TestResult()
        suite = subunit.IsolatedTestSuite()
        sub_suite = unittest.TestSuite()
        sub_suite.addTest(self.SampleTestToIsolate('test_sets_global_state'))
        sub_suite.addTest(self.SampleTestToIsolate('test_sets_global_state'))
        suite.addTest(sub_suite)
        suite.addTest(self.SampleTestToIsolate('test_sets_global_state'))
        suite.run(result)
        self.assertEqual(result.testsRun, 3)
        self.assertEqual(self.SampleTestToIsolate.SETUP, False)
        self.assertEqual(self.SampleTestToIsolate.TEARDOWN, False)
        self.assertEqual(self.SampleTestToIsolate.TEST, False)