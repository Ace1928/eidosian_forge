import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
class TestMultiTenantStoreConnections(base.MultiStoreBaseTest):
    _CONF = cfg.ConfigOpts()

    def setUp(self):
        super(TestMultiTenantStoreConnections, self).setUp()
        enabled_backends = {'swift1': 'swift', 'swift2': 'swift'}
        self.conf = self._CONF
        self.conf(args=[])
        self.conf.register_opt(cfg.DictOpt('enabled_backends'))
        self.config(enabled_backends=enabled_backends)
        store.register_store_opts(self.conf)
        self.config(default_backend='swift1', group='glance_store')
        location.SCHEME_TO_CLS_BACKEND_MAP = {}
        store.create_multi_stores(self.conf)
        self.addCleanup(setattr, location, 'SCHEME_TO_CLS_BACKEND_MAP', dict())
        self.test_dir = self.useFixture(fixtures.TempDir()).path
        self.useFixture(fixtures.MockPatch('swiftclient.Connection', FakeConnection))
        self.context = mock.MagicMock(user='tenant:user1', tenant='tenant', auth_token='0123')
        self.store = swift.MultiTenantStore(self.conf, backend='swift1')
        specs = {'scheme': 'swift', 'auth_or_store_url': 'example.com', 'container': 'cont', 'obj': 'object'}
        self.location = swift.StoreLocation(specs, self.conf, backend_group='swift1')
        self.addCleanup(self.conf.reset)

    def test_basic_connection(self):
        self.store.configure()
        connection = self.store.get_connection(self.location, context=self.context)
        self.assertIsNone(connection.authurl)
        self.assertEqual('1', connection.auth_version)
        self.assertIsNone(connection.user)
        self.assertIsNone(connection.tenant_name)
        self.assertIsNone(connection.key)
        self.assertEqual('https://example.com', connection.preauthurl)
        self.assertEqual('0123', connection.preauthtoken)
        self.assertEqual({}, connection.os_options)

    def test_connection_does_not_use_endpoint_from_catalog(self):
        self.store.configure()
        self.context.service_catalog = [{'endpoint_links': [], 'endpoints': [{'region': 'RegionOne', 'publicURL': 'https://scexample.com'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        connection = self.store.get_connection(self.location, context=self.context)
        self.assertIsNone(connection.authurl)
        self.assertEqual('1', connection.auth_version)
        self.assertIsNone(connection.user)
        self.assertIsNone(connection.tenant_name)
        self.assertIsNone(connection.key)
        self.assertNotEqual('https://scexample.com', connection.preauthurl)
        self.assertEqual('https://example.com', connection.preauthurl)
        self.assertEqual('0123', connection.preauthtoken)
        self.assertEqual({}, connection.os_options)

    def test_connection_manager_does_not_use_endpoint_from_catalog(self):
        self.store.configure()
        self.context.service_catalog = [{'endpoint_links': [], 'endpoints': [{'region': 'RegionOne', 'publicURL': 'https://scexample.com'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        connection_manager = manager.MultiTenantConnectionManager(store=self.store, store_location=self.location, context=self.context)
        conn = connection_manager._init_connection()
        self.assertNotEqual('https://scexample.com', conn.preauthurl)
        self.assertEqual('https://example.com', conn.preauthurl)