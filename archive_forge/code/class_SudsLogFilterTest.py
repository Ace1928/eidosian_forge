import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class SudsLogFilterTest(base.TestCase):
    """Tests for SudsLogFilter."""

    def setUp(self):
        super(SudsLogFilterTest, self).setUp()
        self.log_filter = service.SudsLogFilter()
        self.login = mock.Mock(spec=suds.sax.element.Element)
        self.username = suds.sax.element.Element('username').setText('admin')
        self.password = suds.sax.element.Element('password').setText('password')
        self.session_id = suds.sax.element.Element('session_id').setText('abcdef')

        def login_child_at_path_mock(path):
            if path == 'userName':
                return self.username
            if path == 'password':
                return self.password
            if path == 'sessionID':
                return self.session_id
        self.login.childAtPath.side_effect = login_child_at_path_mock

    def test_filter_with_no_child_at_path(self):
        message = mock.Mock(spec=object)
        record = mock.Mock(msg=message)
        self.assertTrue(self.log_filter.filter(record))

    def test_filter_with_login_failure(self):
        message = mock.Mock(spec=suds.sax.element.Element)

        def child_at_path_mock(path):
            if path == '/Envelope/Body/Login':
                return self.login
        message.childAtPath.side_effect = child_at_path_mock
        record = mock.Mock(msg=message)
        self.assertTrue(self.log_filter.filter(record))
        self.assertEqual('***', self.username.getText())
        self.assertEqual('***', self.password.getText())
        self.assertEqual('bcdef', self.session_id.getText())

    def test_filter_with_session_is_active_failure(self):
        message = mock.Mock(spec=suds.sax.element.Element)

        def child_at_path_mock(path):
            if path == '/Envelope/Body/SessionIsActive':
                return self.login
        message.childAtPath.side_effect = child_at_path_mock
        record = mock.Mock(msg=message)
        self.assertTrue(self.log_filter.filter(record))
        self.assertEqual('***', self.username.getText())
        self.assertEqual('***', self.password.getText())
        self.assertEqual('bcdef', self.session_id.getText())

    def test_filter_with_unknown_failure(self):
        message = mock.Mock(spec=suds.sax.element.Element)

        def child_at_path_mock(path):
            return None
        message.childAtPath.side_effect = child_at_path_mock
        record = mock.Mock(msg=message)
        self.assertTrue(self.log_filter.filter(record))
        self.assertEqual('admin', self.username.getText())
        self.assertEqual('password', self.password.getText())
        self.assertEqual('abcdef', self.session_id.getText())