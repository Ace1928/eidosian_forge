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
class TestTestProtocolServerLostConnection(unittest.TestCase):

    def setUp(self):
        self.client = Python26TestResult()
        self.protocol = subunit.TestProtocolServer(self.client)
        self.test = subunit.RemotedTestCase('old mcdonald')

    def test_lost_connection_no_input(self):
        self.protocol.lostConnection()
        self.assertEqual([], self.client._events)

    def test_lost_connection_after_start(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lostConnection()
        failure = subunit.RemoteError(_u("lost connection during test 'old mcdonald'"))
        self.assertEqual([('startTest', self.test), ('addError', self.test, failure), ('stopTest', self.test)], self.client._events)

    def test_lost_connected_after_error(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('error old mcdonald\n'))
        self.protocol.lostConnection()
        self.assertEqual([('startTest', self.test), ('addError', self.test, subunit.RemoteError(_u(''))), ('stopTest', self.test)], self.client._events)

    def do_connection_lost(self, outcome, opening):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('{} old mcdonald {}'.format(outcome, opening)))
        self.protocol.lostConnection()
        failure = subunit.RemoteError(_u("lost connection during %s report of test 'old mcdonald'") % outcome)
        self.assertEqual([('startTest', self.test), ('addError', self.test, failure), ('stopTest', self.test)], self.client._events)

    def test_lost_connection_during_error(self):
        self.do_connection_lost('error', '[\n')

    def test_lost_connection_during_error_details(self):
        self.do_connection_lost('error', '[ multipart\n')

    def test_lost_connected_after_failure(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('failure old mcdonald\n'))
        self.protocol.lostConnection()
        self.assertEqual([('startTest', self.test), ('addFailure', self.test, subunit.RemoteError(_u(''))), ('stopTest', self.test)], self.client._events)

    def test_lost_connection_during_failure(self):
        self.do_connection_lost('failure', '[\n')

    def test_lost_connection_during_failure_details(self):
        self.do_connection_lost('failure', '[ multipart\n')

    def test_lost_connection_after_success(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('success old mcdonald\n'))
        self.protocol.lostConnection()
        self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)

    def test_lost_connection_during_success(self):
        self.do_connection_lost('success', '[\n')

    def test_lost_connection_during_success_details(self):
        self.do_connection_lost('success', '[ multipart\n')

    def test_lost_connection_during_skip(self):
        self.do_connection_lost('skip', '[\n')

    def test_lost_connection_during_skip_details(self):
        self.do_connection_lost('skip', '[ multipart\n')

    def test_lost_connection_during_xfail(self):
        self.do_connection_lost('xfail', '[\n')

    def test_lost_connection_during_xfail_details(self):
        self.do_connection_lost('xfail', '[ multipart\n')

    def test_lost_connection_during_uxsuccess(self):
        self.do_connection_lost('uxsuccess', '[\n')

    def test_lost_connection_during_uxsuccess_details(self):
        self.do_connection_lost('uxsuccess', '[ multipart\n')