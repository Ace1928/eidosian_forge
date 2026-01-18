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
class TestCredentialStoreRegistry(tests.TestCase):

    def _get_cs_registry(self):
        return config.credential_store_registry

    def test_default_credential_store(self):
        r = self._get_cs_registry()
        default = r.get_credential_store(None)
        self.assertIsInstance(default, config.PlainTextCredentialStore)

    def test_unknown_credential_store(self):
        r = self._get_cs_registry()
        self.assertRaises(KeyError, r.get_credential_store, 'unknown')

    def test_fallback_none_registered(self):
        r = config.CredentialStoreRegistry()
        self.assertEqual(None, r.get_fallback_credentials('http', 'example.com'))

    def test_register(self):
        r = config.CredentialStoreRegistry()
        r.register('stub', StubCredentialStore(), fallback=False)
        r.register('another', StubCredentialStore(), fallback=True)
        self.assertEqual(['another', 'stub'], r.keys())

    def test_register_lazy(self):
        r = config.CredentialStoreRegistry()
        r.register_lazy('stub', 'breezy.tests.test_config', 'StubCredentialStore', fallback=False)
        self.assertEqual(['stub'], r.keys())
        self.assertIsInstance(r.get_credential_store('stub'), StubCredentialStore)

    def test_is_fallback(self):
        r = config.CredentialStoreRegistry()
        r.register('stub1', None, fallback=False)
        r.register('stub2', None, fallback=True)
        self.assertEqual(False, r.is_fallback('stub1'))
        self.assertEqual(True, r.is_fallback('stub2'))

    def test_no_fallback(self):
        r = config.CredentialStoreRegistry()
        store = CountingCredentialStore()
        r.register('count', store, fallback=False)
        self.assertEqual(None, r.get_fallback_credentials('http', 'example.com'))
        self.assertEqual(0, store._calls)

    def test_fallback_credentials(self):
        r = config.CredentialStoreRegistry()
        store = StubCredentialStore()
        store.add_credentials('http', 'example.com', 'somebody', 'geheim')
        r.register('stub', store, fallback=True)
        creds = r.get_fallback_credentials('http', 'example.com')
        self.assertEqual('somebody', creds['user'])
        self.assertEqual('geheim', creds['password'])

    def test_fallback_first_wins(self):
        r = config.CredentialStoreRegistry()
        stub1 = StubCredentialStore()
        stub1.add_credentials('http', 'example.com', 'somebody', 'stub1')
        r.register('stub1', stub1, fallback=True)
        stub2 = StubCredentialStore()
        stub2.add_credentials('http', 'example.com', 'somebody', 'stub2')
        r.register('stub2', stub1, fallback=True)
        creds = r.get_fallback_credentials('http', 'example.com')
        self.assertEqual('somebody', creds['user'])
        self.assertEqual('stub1', creds['password'])