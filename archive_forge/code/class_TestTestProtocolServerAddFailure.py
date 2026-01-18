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
class TestTestProtocolServerAddFailure(unittest.TestCase):

    def setUp(self):
        self.client = ExtendedTestResult()
        self.protocol = subunit.TestProtocolServer(self.client)
        self.protocol.lineReceived(_b('test mcdonalds farm\n'))
        self.test = subunit.RemotedTestCase('mcdonalds farm')

    def assertFailure(self, details):
        self.assertEqual([('startTest', self.test), ('addFailure', self.test, details), ('stopTest', self.test)], self.client._events)

    def simple_failure_keyword(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm\n' % keyword))
        details = {}
        self.assertFailure(details)

    def test_simple_failure(self):
        self.simple_failure_keyword('failure')

    def test_simple_failure_colon(self):
        self.simple_failure_keyword('failure:')

    def test_failure_empty_message(self):
        self.protocol.lineReceived(_b('failure mcdonalds farm [\n'))
        self.protocol.lineReceived(_b(']\n'))
        details = {}
        details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b('')])
        self.assertFailure(details)

    def failure_quoted_bracket(self, keyword):
        self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        details = {}
        details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b(']\n')])
        self.assertFailure(details)

    def test_failure_quoted_bracket(self):
        self.failure_quoted_bracket('failure')

    def test_failure_colon_quoted_bracket(self):
        self.failure_quoted_bracket('failure:')