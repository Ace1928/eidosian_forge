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
class TestSingleTenantStoreConnections(base.MultiStoreBaseTest):
    _CONF = cfg.ConfigOpts()

    def setUp(self):
        super(TestSingleTenantStoreConnections, self).setUp()
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
        self.store = swift.SingleTenantStore(self.conf, backend='swift1')
        self.store.configure()
        specs = {'scheme': 'swift', 'auth_or_store_url': 'example.com/v2/', 'user': 'tenant:user1', 'key': 'key1', 'container': 'cont', 'obj': 'object'}
        self.location = swift.StoreLocation(specs, self.conf, backend_group='swift1')
        self.register_store_backend_schemes(self.store, 'swift', 'swift1')
        self.addCleanup(self.conf.reset)

    def test_basic_connection(self):
        connection = self.store.get_connection(self.location)
        self.assertEqual('https://example.com/v2/', connection.authurl)
        self.assertEqual('2', connection.auth_version)
        self.assertEqual('user1', connection.user)
        self.assertEqual('tenant', connection.tenant_name)
        self.assertEqual('key1', connection.key)
        self.assertIsNone(connection.preauthurl)
        self.assertFalse(connection.insecure)
        self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL'}, connection.os_options)

    def test_connection_with_conf_endpoint(self):
        ctx = mock.MagicMock(user='tenant:user1', tenant='tenant')
        self.config(group='swift1', swift_store_endpoint='https://internal.com')
        self.store.configure()
        connection = self.store.get_connection(self.location, context=ctx)
        self.assertEqual('https://example.com/v2/', connection.authurl)
        self.assertEqual('2', connection.auth_version)
        self.assertEqual('user1', connection.user)
        self.assertEqual('tenant', connection.tenant_name)
        self.assertEqual('key1', connection.key)
        self.assertEqual('https://internal.com', connection.preauthurl)
        self.assertFalse(connection.insecure)
        self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL'}, connection.os_options)

    def test_connection_with_conf_endpoint_no_context(self):
        self.config(group='swift1', swift_store_endpoint='https://internal.com')
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertEqual('https://example.com/v2/', connection.authurl)
        self.assertEqual('2', connection.auth_version)
        self.assertEqual('user1', connection.user)
        self.assertEqual('tenant', connection.tenant_name)
        self.assertEqual('key1', connection.key)
        self.assertEqual('https://internal.com', connection.preauthurl)
        self.assertFalse(connection.insecure)
        self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL'}, connection.os_options)

    def test_connection_with_no_trailing_slash(self):
        self.location.auth_or_store_url = 'example.com/v2'
        connection = self.store.get_connection(self.location)
        self.assertEqual('https://example.com/v2/', connection.authurl)

    def test_connection_insecure(self):
        self.config(group='swift1', swift_store_auth_insecure=True)
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertTrue(connection.insecure)

    def test_connection_with_auth_v1(self):
        self.config(group='swift1', swift_store_auth_version='1')
        self.store.configure()
        self.location.user = 'auth_v1_user'
        connection = self.store.get_connection(self.location)
        self.assertEqual('1', connection.auth_version)
        self.assertEqual('auth_v1_user', connection.user)
        self.assertIsNone(connection.tenant_name)

    def test_connection_invalid_user(self):
        self.store.configure()
        self.location.user = 'invalid:format:user'
        self.assertRaises(exceptions.BadStoreUri, self.store.get_connection, self.location)

    def test_connection_missing_user(self):
        self.store.configure()
        self.location.user = None
        self.assertRaises(exceptions.BadStoreUri, self.store.get_connection, self.location)

    def test_connection_with_region(self):
        self.config(group='swift1', swift_store_region='Sahara')
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertEqual({'region_name': 'Sahara', 'service_type': 'object-store', 'endpoint_type': 'publicURL'}, connection.os_options)

    def test_connection_with_service_type(self):
        self.config(group='swift1', swift_store_service_type='shoe-store')
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertEqual({'service_type': 'shoe-store', 'endpoint_type': 'publicURL'}, connection.os_options)

    def test_connection_with_endpoint_type(self):
        self.config(group='swift1', swift_store_endpoint_type='internalURL')
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'internalURL'}, connection.os_options)

    def test_bad_location_uri(self):
        self.store.configure()
        self.location.uri = 'http://bad_uri://'
        self.assertRaises(exceptions.BadStoreUri, self.location.parse_uri, self.location.uri)

    def test_bad_location_uri_invalid_credentials(self):
        self.store.configure()
        self.location.uri = 'swift://bad_creds@uri/cont/obj'
        self.assertRaises(exceptions.BadStoreUri, self.location.parse_uri, self.location.uri)

    def test_bad_location_uri_invalid_object_path(self):
        self.store.configure()
        self.location.uri = 'swift://user:key@uri/cont'
        self.assertRaises(exceptions.BadStoreUri, self.location.parse_uri, self.location.uri)

    def test_ref_overrides_defaults(self):
        self.config(group='swift1', swift_store_auth_version='2', swift_store_user='testuser', swift_store_key='testpass', swift_store_auth_address='testaddress', swift_store_endpoint_type='internalURL', swift_store_config_file='somefile')
        self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'auth_version': '3', 'user': 'user:pass', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}}
        self.store.configure()
        self.assertEqual('user:pass', self.store.user)
        self.assertEqual('3', self.store.auth_version)
        self.assertEqual('authurl.com', self.store.auth_address)
        self.assertEqual('default', self.store.user_domain_id)
        self.assertEqual('ignored', self.store.user_domain_name)
        self.assertEqual('default', self.store.project_domain_id)
        self.assertEqual('ignored', self.store.project_domain_name)

    def test_with_v3_auth(self):
        self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'auth_version': '3', 'user': 'user:pass', 'key': 'password', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}}
        self.store.configure()
        connection = self.store.get_connection(self.location)
        self.assertEqual('3', connection.auth_version)
        self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL', 'user_domain_id': 'default', 'user_domain_name': 'ignored', 'project_domain_id': 'default', 'project_domain_name': 'ignored'}, connection.os_options)