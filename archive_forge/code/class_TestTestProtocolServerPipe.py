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
class TestTestProtocolServerPipe(unittest.TestCase):

    def test_story(self):
        client = unittest.TestResult()
        protocol = subunit.TestProtocolServer(client)
        traceback = 'foo.c:53:ERROR invalid state\n'
        pipe = BytesIO(_b('test old mcdonald\nsuccess old mcdonald\ntest bing crosby\nfailure bing crosby [\n' + traceback + ']\ntest an error\nerror an error\n'))
        protocol.readFrom(pipe)
        bing = subunit.RemotedTestCase('bing crosby')
        an_error = subunit.RemotedTestCase('an error')
        self.assertEqual(client.errors, [(an_error, _remote_exception_repr + '\n')])
        self.assertEqual(client.failures, [(bing, _remote_exception_repr + ': ' + details_to_str({'traceback': text_content(traceback)}) + '\n')])
        self.assertEqual(client.testsRun, 3)

    def test_non_test_characters_forwarded_immediately(self):
        pass