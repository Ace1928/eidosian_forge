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
class TestInTestMultipart(unittest.TestCase):

    def setUp(self):
        self.client = ExtendedTestResult()
        self.protocol = subunit.TestProtocolServer(self.client)
        self.protocol.lineReceived(_b('test mcdonalds farm\n'))
        self.test = subunit.RemotedTestCase(_u('mcdonalds farm'))

    def test__outcome_sets_details_parser(self):
        self.protocol._reading_success_details.details_parser = None
        self.protocol._state._outcome(0, _b('mcdonalds farm [ multipart\n'), None, self.protocol._reading_success_details)
        parser = self.protocol._reading_success_details.details_parser
        self.assertNotEqual(None, parser)
        self.assertIsInstance(parser, subunit.details.MultipartDetailsParser)