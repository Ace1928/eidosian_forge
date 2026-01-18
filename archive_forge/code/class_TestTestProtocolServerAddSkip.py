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
class TestTestProtocolServerAddSkip(unittest.TestCase):
    """Tests for the skip keyword.

    In Python this meets the testtools extended TestResult contract.
    (See https://launchpad.net/testtools).
    """

    def setUp(self):
        """Setup a test object ready to be skipped."""
        self.client = ExtendedTestResult()
        self.protocol = subunit.TestProtocolServer(self.client)
        self.protocol.lineReceived(_b('test mcdonalds farm\n'))
        self.test = self.client._events[-1][-1]

    def assertSkip(self, reason):
        details = {}
        if reason is not None:
            details['reason'] = Content(ContentType('text', 'plain'), lambda: [reason])
        self.assertEqual([('startTest', self.test), ('addSkip', self.test, details), ('stopTest', self.test)], self.client._events)

    def simple_skip_keyword(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm\n' % keyword))
        self.assertSkip(None)

    def test_simple_skip(self):
        self.simple_skip_keyword('skip')

    def test_simple_skip_colon(self):
        self.simple_skip_keyword('skip:')

    def test_skip_empty_message(self):
        self.protocol.lineReceived(_b('skip mcdonalds farm [\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.assertSkip(_b(''))

    def skip_quoted_bracket(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.assertSkip(_b(']\n'))

    def test_skip_quoted_bracket(self):
        self.skip_quoted_bracket('skip')

    def test_skip_colon_quoted_bracket(self):
        self.skip_quoted_bracket('skip:')