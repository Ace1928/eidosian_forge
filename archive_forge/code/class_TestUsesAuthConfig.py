import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
class TestUsesAuthConfig(TestCaseWithSFTPServer):
    """Test that AuthenticationConfig can supply default usernames."""

    def get_transport_for_connection(self, set_config):
        port = self.get_server().port
        if set_config:
            conf = config.AuthenticationConfig()
            conf._get_config().update({'sftptest': {'scheme': 'ssh', 'port': port, 'user': 'bar'}})
            conf._save()
        t = _mod_transport.get_transport_from_url('sftp://localhost:%d' % port)
        t.has('foo')
        return t

    def test_sftp_uses_config(self):
        t = self.get_transport_for_connection(set_config=True)
        self.assertEqual('bar', t._get_credentials()[0])

    def test_sftp_is_none_if_no_config(self):
        t = self.get_transport_for_connection(set_config=False)
        self.assertIs(None, t._get_credentials()[0])

    def test_sftp_doesnt_prompt_username(self):
        ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
        t = self.get_transport_for_connection(set_config=False)
        self.assertIs(None, t._get_credentials()[0])
        self.assertEqual('', ui.ui_factory.stdout.getvalue())
        self.assertEqual(0, ui.ui_factory.stdin.tell())