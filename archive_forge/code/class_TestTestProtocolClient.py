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
class TestTestProtocolClient(TestCase):

    def setUp(self):
        super().setUp()
        self.io = BytesIO()
        self.protocol = subunit.TestProtocolClient(self.io)
        self.unicode_test = PlaceHolder(_u('☃'))
        self.test = TestTestProtocolClient('test_start_test')
        self.sample_details = {'something': Content(ContentType('text', 'plain'), lambda: [_b('serialised\nform')])}
        self.sample_tb_details = dict(self.sample_details)
        self.sample_tb_details['traceback'] = TracebackContent(subunit.RemoteError(_u('boo qux')), self.test)

    def test_start_test(self):
        """Test startTest on a TestProtocolClient."""
        self.protocol.startTest(self.test)
        self.assertEqual(self.io.getvalue(), _b('test: %s\n' % self.test.id()))

    def test_start_test_unicode_id(self):
        """Test startTest on a TestProtocolClient."""
        self.protocol.startTest(self.unicode_test)
        expected = _b('test: ') + _u('☃').encode('utf8') + _b('\n')
        self.assertEqual(expected, self.io.getvalue())

    def test_stop_test(self):
        self.protocol.stopTest(self.test)
        self.assertEqual(self.io.getvalue(), _b(''))

    def test_add_success(self):
        """Test addSuccess on a TestProtocolClient."""
        self.protocol.addSuccess(self.test)
        self.assertEqual(self.io.getvalue(), _b('successful: %s\n' % self.test.id()))

    def test_add_outcome_unicode_id(self):
        """Test addSuccess on a TestProtocolClient."""
        self.protocol.addSuccess(self.unicode_test)
        expected = _b('successful: ') + _u('☃').encode('utf8') + _b('\n')
        self.assertEqual(expected, self.io.getvalue())

    def test_add_success_details(self):
        """Test addSuccess on a TestProtocolClient with details."""
        self.protocol.addSuccess(self.test, details=self.sample_details)
        self.assertEqual(self.io.getvalue(), _b('successful: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\n]\n' % self.test.id()))

    def test_add_failure(self):
        """Test addFailure on a TestProtocolClient."""
        self.protocol.addFailure(self.test, subunit.RemoteError(_u('boo qux')))
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('failure: %s [\n' + _remote_exception_str + ': boo qux\n' + ']\n') % self.test.id())), Equals(_b(('failure: %s [\n' + _remote_exception_repr + ': boo qux\n' + ']\n') % self.test.id()))))

    def test_add_failure_details(self):
        """Test addFailure on a TestProtocolClient with details."""
        self.protocol.addFailure(self.test, details=self.sample_tb_details)
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('failure: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_str_chunked + ']\n') % self.test.id())), Equals(_b(('failure: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_repr_chunked + ']\n') % self.test.id()))))

    def test_add_error(self):
        """Test stopTest on a TestProtocolClient."""
        self.protocol.addError(self.test, subunit.RemoteError(_u('phwoar crikey')))
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('error: %s [\n' + _remote_exception_str + ': phwoar crikey\n]\n') % self.test.id())), Equals(_b(('error: %s [\n' + _remote_exception_repr + ': phwoar crikey\n]\n') % self.test.id()))))

    def test_add_error_details(self):
        """Test stopTest on a TestProtocolClient with details."""
        self.protocol.addError(self.test, details=self.sample_tb_details)
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('error: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_str_chunked + ']\n') % self.test.id())), Equals(_b(('error: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_repr_chunked + ']\n') % self.test.id()))))

    def test_add_expected_failure(self):
        """Test addExpectedFailure on a TestProtocolClient."""
        self.protocol.addExpectedFailure(self.test, subunit.RemoteError(_u('phwoar crikey')))
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('xfail: %s [\n' + _remote_exception_str + ': phwoar crikey\n]\n') % self.test.id())), Equals(_b(('xfail: %s [\n' + _remote_exception_repr + ': phwoar crikey\n]\n') % self.test.id()))))

    def test_add_expected_failure_details(self):
        """Test addExpectedFailure on a TestProtocolClient with details."""
        self.protocol.addExpectedFailure(self.test, details=self.sample_tb_details)
        self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('xfail: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_str_chunked + ']\n') % self.test.id())), Equals(_b(('xfail: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\nContent-Type: text/x-traceback;charset=utf8,language=python\ntraceback\n' + _remote_exception_repr_chunked + ']\n') % self.test.id()))))

    def test_add_skip(self):
        """Test addSkip on a TestProtocolClient."""
        self.protocol.addSkip(self.test, 'Has it really?')
        self.assertEqual(self.io.getvalue(), _b('skip: %s [\nHas it really?\n]\n' % self.test.id()))

    def test_add_skip_details(self):
        """Test addSkip on a TestProtocolClient with details."""
        details = {'reason': Content(ContentType('text', 'plain'), lambda: [_b('Has it really?')])}
        self.protocol.addSkip(self.test, details=details)
        self.assertEqual(self.io.getvalue(), _b('skip: %s [ multipart\nContent-Type: text/plain\nreason\nE\r\nHas it really?0\r\n]\n' % self.test.id()))

    def test_progress_set(self):
        self.protocol.progress(23, subunit.PROGRESS_SET)
        self.assertEqual(self.io.getvalue(), _b('progress: 23\n'))

    def test_progress_neg_cur(self):
        self.protocol.progress(-23, subunit.PROGRESS_CUR)
        self.assertEqual(self.io.getvalue(), _b('progress: -23\n'))

    def test_progress_pos_cur(self):
        self.protocol.progress(23, subunit.PROGRESS_CUR)
        self.assertEqual(self.io.getvalue(), _b('progress: +23\n'))

    def test_progress_pop(self):
        self.protocol.progress(1234, subunit.PROGRESS_POP)
        self.assertEqual(self.io.getvalue(), _b('progress: pop\n'))

    def test_progress_push(self):
        self.protocol.progress(1234, subunit.PROGRESS_PUSH)
        self.assertEqual(self.io.getvalue(), _b('progress: push\n'))

    def test_time(self):
        self.protocol.time(datetime.datetime(2009, 10, 11, 12, 13, 14, 15, iso8601.UTC))
        self.assertEqual(_b('time: 2009-10-11 12:13:14.000015Z\n'), self.io.getvalue())

    def test_add_unexpected_success(self):
        """Test addUnexpectedSuccess on a TestProtocolClient."""
        self.protocol.addUnexpectedSuccess(self.test)
        self.assertEqual(self.io.getvalue(), _b('uxsuccess: %s\n' % self.test.id()))

    def test_add_unexpected_success_details(self):
        """Test addUnexpectedSuccess on a TestProtocolClient with details."""
        self.protocol.addUnexpectedSuccess(self.test, details=self.sample_details)
        self.assertEqual(self.io.getvalue(), _b('uxsuccess: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\n]\n' % self.test.id()))

    def test_tags_empty(self):
        self.protocol.tags(set(), set())
        self.assertEqual(_b(''), self.io.getvalue())

    def test_tags_add(self):
        self.protocol.tags({'foo'}, set())
        self.assertEqual(_b('tags: foo\n'), self.io.getvalue())

    def test_tags_both(self):
        self.protocol.tags({'quux'}, {'bar'})
        self.assertThat([b'tags: quux -bar\n', b'tags: -bar quux\n'], Contains(self.io.getvalue()))

    def test_tags_gone(self):
        self.protocol.tags(set(), {'bar'})
        self.assertEqual(_b('tags: -bar\n'), self.io.getvalue())