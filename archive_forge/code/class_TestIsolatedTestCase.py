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
class TestIsolatedTestCase(TestCase):

    class SampleIsolatedTestCase(subunit.IsolatedTestCase):
        SETUP = False
        TEARDOWN = False
        TEST = False

        def setUp(self):
            TestIsolatedTestCase.SampleIsolatedTestCase.SETUP = True

        def tearDown(self):
            TestIsolatedTestCase.SampleIsolatedTestCase.TEARDOWN = True

        def test_sets_global_state(self):
            TestIsolatedTestCase.SampleIsolatedTestCase.TEST = True

    def test_construct(self):
        self.SampleIsolatedTestCase('test_sets_global_state')

    @skipIf(os.name != 'posix', 'Need a posix system for forking tests')
    def test_run(self):
        result = unittest.TestResult()
        test = self.SampleIsolatedTestCase('test_sets_global_state')
        test.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(self.SampleIsolatedTestCase.SETUP, False)
        self.assertEqual(self.SampleIsolatedTestCase.TEARDOWN, False)
        self.assertEqual(self.SampleIsolatedTestCase.TEST, False)

    def test_debug(self):
        pass