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
class TestMultipleContainers(base.MultiStoreBaseTest):
    _CONF = cfg.ConfigOpts()

    def setUp(self):
        super(TestMultipleContainers, self).setUp()
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
        self.config(group='swift1', swift_store_multiple_containers_seed=3)
        self.store = swift.SingleTenantStore(self.conf, backend='swift1')
        self.store.configure()
        self.register_store_backend_schemes(self.store, 'swift', 'swift1')
        self.addCleanup(self.conf.reset)

    def test_get_container_name_happy_path_with_seed_three(self):
        test_image_id = 'fdae39a1-bac5-4238-aba4-69bcc726e848'
        actual = self.store.get_container_name(test_image_id, 'default_container')
        expected = 'default_container_fda'
        self.assertEqual(expected, actual)

    def test_get_container_name_with_negative_seed(self):
        self.assertRaises(ValueError, self.config, group='swift1', swift_store_multiple_containers_seed=-1)

    def test_get_container_name_with_seed_beyond_max(self):
        self.assertRaises(ValueError, self.config, group='swift1', swift_store_multiple_containers_seed=33)

    def test_get_container_name_with_max_seed(self):
        self.config(group='swift1', swift_store_multiple_containers_seed=32)
        self.store = swift.SingleTenantStore(self.conf, backend='swift1')
        test_image_id = 'fdae39a1-bac5-4238-aba4-69bcc726e848'
        actual = self.store.get_container_name(test_image_id, 'default_container')
        expected = 'default_container_' + test_image_id
        self.assertEqual(expected, actual)

    def test_get_container_name_with_dash(self):
        self.config(group='swift1', swift_store_multiple_containers_seed=10)
        self.store = swift.SingleTenantStore(self.conf, backend='swift1')
        test_image_id = 'fdae39a1-bac5-4238-aba4-69bcc726e848'
        actual = self.store.get_container_name(test_image_id, 'default_container')
        expected = 'default_container_' + 'fdae39a1-ba'
        self.assertEqual(expected, actual)

    def test_get_container_name_with_min_seed(self):
        self.config(group='swift1', swift_store_multiple_containers_seed=1)
        self.store = swift.SingleTenantStore(self.conf, backend='swift1')
        test_image_id = 'fdae39a1-bac5-4238-aba4-69bcc726e848'
        actual = self.store.get_container_name(test_image_id, 'default_container')
        expected = 'default_container_' + 'f'
        self.assertEqual(expected, actual)

    def test_get_container_name_with_multiple_containers_turned_off(self):
        self.config(group='swift1', swift_store_multiple_containers_seed=0)
        self.store.configure()
        test_image_id = 'random_id'
        actual = self.store.get_container_name(test_image_id, 'default_container')
        expected = 'default_container'
        self.assertEqual(expected, actual)