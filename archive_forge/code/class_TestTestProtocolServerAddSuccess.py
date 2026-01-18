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
class TestTestProtocolServerAddSuccess(unittest.TestCase):

    def setUp(self):
        self.client = ExtendedTestResult()
        self.protocol = subunit.TestProtocolServer(self.client)
        self.protocol.lineReceived(_b('test mcdonalds farm\n'))
        self.test = subunit.RemotedTestCase('mcdonalds farm')

    def simple_success_keyword(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm\n' % keyword))
        self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)

    def test_simple_success(self):
        self.simple_success_keyword('successful')

    def test_simple_success_colon(self):
        self.simple_success_keyword('successful:')

    def assertSuccess(self, details):
        self.assertEqual([('startTest', self.test), ('addSuccess', self.test, details), ('stopTest', self.test)], self.client._events)

    def test_success_empty_message(self):
        self.protocol.lineReceived(_b('success mcdonalds farm [\n'))
        self.protocol.lineReceived(_b(']\n'))
        details = {}
        details['message'] = Content(ContentType('text', 'plain'), lambda: [_b('')])
        self.assertSuccess(details)

    def success_quoted_bracket(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        details = {}
        details['message'] = Content(ContentType('text', 'plain'), lambda: [_b(']\n')])
        self.assertSuccess(details)

    def test_success_quoted_bracket(self):
        self.success_quoted_bracket('success')

    def test_success_colon_quoted_bracket(self):
        self.success_quoted_bracket('success:')