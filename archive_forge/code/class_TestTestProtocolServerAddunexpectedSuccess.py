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
class TestTestProtocolServerAddunexpectedSuccess(TestCase):
    """Tests for the uxsuccess keyword."""

    def capture_expected_failure(self, test, err):
        self._events.append((test, err))

    def setup_python26(self):
        """Setup a test object ready to be xfailed and thunk to success."""
        self.client = Python26TestResult()
        self.setup_protocol()

    def setup_python27(self):
        """Setup a test object ready to be xfailed."""
        self.client = Python27TestResult()
        self.setup_protocol()

    def setup_python_ex(self):
        """Setup a test object ready to be xfailed with details."""
        self.client = ExtendedTestResult()
        self.setup_protocol()

    def setup_protocol(self):
        """Setup the protocol based on self.client."""
        self.protocol = subunit.TestProtocolServer(self.client)
        self.protocol.lineReceived(_b('test mcdonalds farm\n'))
        self.test = self.client._events[-1][-1]

    def simple_uxsuccess_keyword(self, keyword, as_fail):
        self.protocol.lineReceived(_b('%s mcdonalds farm\n' % keyword))
        self.check_fail_or_uxsuccess(as_fail)

    def check_fail_or_uxsuccess(self, as_fail, error_message=None):
        details = {}
        if error_message is not None:
            details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b(error_message)])
        if isinstance(self.client, ExtendedTestResult):
            value = details
        else:
            value = None
        if as_fail:
            self.client._events[1] = self.client._events[1][:2]
            self.assertEqual([('startTest', self.test), ('addFailure', self.test), ('stopTest', self.test)], self.client._events)
        elif value:
            self.assertEqual([('startTest', self.test), ('addUnexpectedSuccess', self.test, value), ('stopTest', self.test)], self.client._events)
        else:
            self.assertEqual([('startTest', self.test), ('addUnexpectedSuccess', self.test), ('stopTest', self.test)], self.client._events)

    def test_simple_uxsuccess(self):
        self.setup_python26()
        self.simple_uxsuccess_keyword('uxsuccess', True)
        self.setup_python27()
        self.simple_uxsuccess_keyword('uxsuccess', False)
        self.setup_python_ex()
        self.simple_uxsuccess_keyword('uxsuccess', False)

    def test_simple_uxsuccess_colon(self):
        self.setup_python26()
        self.simple_uxsuccess_keyword('uxsuccess:', True)
        self.setup_python27()
        self.simple_uxsuccess_keyword('uxsuccess:', False)
        self.setup_python_ex()
        self.simple_uxsuccess_keyword('uxsuccess:', False)

    def test_uxsuccess_empty_message(self):
        self.setup_python26()
        self.empty_message(True)
        self.setup_python27()
        self.empty_message(False)
        self.setup_python_ex()
        self.empty_message(False, error_message='')

    def empty_message(self, as_fail, error_message='\n'):
        self.protocol.lineReceived(_b('uxsuccess mcdonalds farm [\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.check_fail_or_uxsuccess(as_fail, error_message)

    def uxsuccess_quoted_bracket(self, keyword, as_fail):
        self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.check_fail_or_uxsuccess(as_fail, ']\n')

    def test_uxsuccess_quoted_bracket(self):
        self.setup_python26()
        self.uxsuccess_quoted_bracket('uxsuccess', True)
        self.setup_python27()
        self.uxsuccess_quoted_bracket('uxsuccess', False)
        self.setup_python_ex()
        self.uxsuccess_quoted_bracket('uxsuccess', False)

    def test_uxsuccess_colon_quoted_bracket(self):
        self.setup_python26()
        self.uxsuccess_quoted_bracket('uxsuccess:', True)
        self.setup_python27()
        self.uxsuccess_quoted_bracket('uxsuccess:', False)
        self.setup_python_ex()
        self.uxsuccess_quoted_bracket('uxsuccess:', False)