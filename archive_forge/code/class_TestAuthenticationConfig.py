import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestAuthenticationConfig(tests.TestCaseInTempDir):
    """Test AuthenticationConfig behaviour"""

    def _check_default_password_prompt(self, expected_prompt_format, scheme, host=None, port=None, realm=None, path=None):
        if host is None:
            host = 'bar.org'
        user, password = ('jim', 'precious')
        expected_prompt = expected_prompt_format % {'scheme': scheme, 'host': host, 'port': port, 'user': user, 'realm': realm}
        ui.ui_factory = tests.TestUIFactory(stdin=password + '\n')
        conf = config.AuthenticationConfig()
        self.assertEqual(password, conf.get_password(scheme, host, user, port=port, realm=realm, path=path))
        self.assertEqual(expected_prompt, ui.ui_factory.stderr.getvalue())
        self.assertEqual('', ui.ui_factory.stdout.getvalue())

    def _check_default_username_prompt(self, expected_prompt_format, scheme, host=None, port=None, realm=None, path=None):
        if host is None:
            host = 'bar.org'
        username = 'jim'
        expected_prompt = expected_prompt_format % {'scheme': scheme, 'host': host, 'port': port, 'realm': realm}
        ui.ui_factory = tests.TestUIFactory(stdin=username + '\n')
        conf = config.AuthenticationConfig()
        self.assertEqual(username, conf.get_user(scheme, host, port=port, realm=realm, path=path, ask=True))
        self.assertEqual(expected_prompt, ui.ui_factory.stderr.getvalue())
        self.assertEqual('', ui.ui_factory.stdout.getvalue())

    def test_username_defaults_prompts(self):
        self._check_default_username_prompt('FTP %(host)s username: ', 'ftp')
        self._check_default_username_prompt('FTP %(host)s:%(port)d username: ', 'ftp', port=10020)
        self._check_default_username_prompt('SSH %(host)s:%(port)d username: ', 'ssh', port=12345)

    def test_username_default_no_prompt(self):
        conf = config.AuthenticationConfig()
        self.assertEqual(None, conf.get_user('ftp', 'example.com'))
        self.assertEqual('explicitdefault', conf.get_user('ftp', 'example.com', default='explicitdefault'))

    def test_password_default_prompts(self):
        self._check_default_password_prompt('FTP %(user)s@%(host)s password: ', 'ftp')
        self._check_default_password_prompt('FTP %(user)s@%(host)s:%(port)d password: ', 'ftp', port=10020)
        self._check_default_password_prompt('SSH %(user)s@%(host)s:%(port)d password: ', 'ssh', port=12345)
        self._check_default_password_prompt('SMTP %(user)s@%(host)s password: ', 'smtp')
        self._check_default_password_prompt('SMTP %(user)s@%(host)s password: ', 'smtp', host='bar.org:10025')
        self._check_default_password_prompt('SMTP %(user)s@%(host)s:%(port)d password: ', 'smtp', port=10025)

    def test_ssh_password_emits_warning(self):
        conf = config.AuthenticationConfig(_file=BytesIO(b'\n[ssh with password]\nscheme=ssh\nhost=bar.org\nuser=jim\npassword=jimpass\n'))
        entered_password = 'typed-by-hand'
        ui.ui_factory = tests.TestUIFactory(stdin=entered_password + '\n')
        self.assertEqual(entered_password, conf.get_password('ssh', 'bar.org', user='jim'))
        self.assertContainsRe(self.get_log(), 'password ignored in section \\[ssh with password\\]')

    def test_ssh_without_password_doesnt_emit_warning(self):
        conf = config.AuthenticationConfig(_file=BytesIO(b'\n[ssh with password]\nscheme=ssh\nhost=bar.org\nuser=jim\n'))
        entered_password = 'typed-by-hand'
        ui.ui_factory = tests.TestUIFactory(stdin=entered_password + '\n')
        self.assertEqual(entered_password, conf.get_password('ssh', 'bar.org', user='jim'))
        self.assertNotContainsRe(self.get_log(), 'password ignored in section \\[ssh with password\\]')

    def test_uses_fallback_stores(self):
        self.overrideAttr(config, 'credential_store_registry', config.CredentialStoreRegistry())
        store = StubCredentialStore()
        store.add_credentials('http', 'example.com', 'joe', 'secret')
        config.credential_store_registry.register('stub', store, fallback=True)
        conf = config.AuthenticationConfig(_file=BytesIO())
        creds = conf.get_credentials('http', 'example.com')
        self.assertEqual('joe', creds['user'])
        self.assertEqual('secret', creds['password'])