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