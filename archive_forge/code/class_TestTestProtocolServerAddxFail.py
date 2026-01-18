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
class TestTestProtocolServerAddxFail(unittest.TestCase):
    """Tests for the xfail keyword.

    In Python this can thunk through to Success due to stdlib limitations (see
    README).
    """

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

    def simple_xfail_keyword(self, keyword, as_success):
        self.protocol.lineReceived(_b('%s mcdonalds farm\n' % keyword))
        self.check_success_or_xfail(as_success)

    def check_success_or_xfail(self, as_success, error_message=None):
        if as_success:
            self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)
        else:
            details = {}
            if error_message is not None:
                details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b(error_message)])
            if isinstance(self.client, ExtendedTestResult):
                value = details
            elif error_message is not None:
                value = subunit.RemoteError(details_to_str(details))
            else:
                value = subunit.RemoteError()
            self.assertEqual([('startTest', self.test), ('addExpectedFailure', self.test, value), ('stopTest', self.test)], self.client._events)

    def test_simple_xfail(self):
        self.setup_python26()
        self.simple_xfail_keyword('xfail', True)
        self.setup_python27()
        self.simple_xfail_keyword('xfail', False)
        self.setup_python_ex()
        self.simple_xfail_keyword('xfail', False)

    def test_simple_xfail_colon(self):
        self.setup_python26()
        self.simple_xfail_keyword('xfail:', True)
        self.setup_python27()
        self.simple_xfail_keyword('xfail:', False)
        self.setup_python_ex()
        self.simple_xfail_keyword('xfail:', False)

    def test_xfail_empty_message(self):
        self.setup_python26()
        self.empty_message(True)
        self.setup_python27()
        self.empty_message(False)
        self.setup_python_ex()
        self.empty_message(False, error_message='')

    def empty_message(self, as_success, error_message='\n'):
        self.protocol.lineReceived(_b('xfail mcdonalds farm [\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.check_success_or_xfail(as_success, error_message)

    def xfail_quoted_bracket(self, keyword, as_success):
        self.protocol.lineReceived(_b('%s mcdonalds farm [\n' % keyword))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.check_success_or_xfail(as_success, ']\n')

    def test_xfail_quoted_bracket(self):
        self.setup_python26()
        self.xfail_quoted_bracket('xfail', True)
        self.setup_python27()
        self.xfail_quoted_bracket('xfail', False)
        self.setup_python_ex()
        self.xfail_quoted_bracket('xfail', False)

    def test_xfail_colon_quoted_bracket(self):
        self.setup_python26()
        self.xfail_quoted_bracket('xfail:', True)
        self.setup_python27()
        self.xfail_quoted_bracket('xfail:', False)
        self.setup_python_ex()
        self.xfail_quoted_bracket('xfail:', False)