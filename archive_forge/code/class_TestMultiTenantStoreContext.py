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
class TestMultiTenantStoreContext(base.MultiStoreBaseTest):
    _CONF = cfg.ConfigOpts()

    def setUp(self):
        """Establish a clean test environment."""
        super(TestMultiTenantStoreContext, self).setUp()
        config = SWIFT_CONF.copy()
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
        self.store = Store(self.conf, backend='swift1')
        self.config(group='swift1', **config)
        self.store.configure()
        self.register_store_backend_schemes(self.store, 'swift', 'swift1')
        service_catalog = [{'endpoint_links': [], 'endpoints': [{'region': 'RegionOne', 'publicURL': 'http://127.0.0.1:0'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        self.ctx = mock.MagicMock(service_catalog=service_catalog, user='tenant:user1', tenant='tenant', auth_token='0123')
        self.addCleanup(self.conf.reset)

    @requests_mock.mock()
    def test_download_context(self, m):
        """Verify context (ie token) is passed to swift on download."""
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        uri = 'swift+http://127.0.0.1/glance_123/123'
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        m.get('http://127.0.0.1/glance_123/123', headers={'Content-Length': '0'})
        store.get(loc, context=self.ctx)
        self.assertEqual(b'0123', m.last_request.headers['X-Auth-Token'])

    @requests_mock.mock()
    def test_upload_context(self, m):
        """Verify context (ie token) is passed to swift on upload."""
        head_req = m.head('http://127.0.0.1/glance_123', text='Some data', status_code=201)
        put_req = m.put('http://127.0.0.1/glance_123/123')
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        content = b'Some data'
        pseudo_file = io.BytesIO(content)
        store.add('123', pseudo_file, len(content), context=self.ctx)
        self.assertEqual(b'0123', head_req.last_request.headers['X-Auth-Token'])
        self.assertEqual(b'0123', put_req.last_request.headers['X-Auth-Token'])