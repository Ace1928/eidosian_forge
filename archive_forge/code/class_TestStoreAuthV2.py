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
class TestStoreAuthV2(TestStoreAuthV1):

    def getConfig(self):
        config = super(TestStoreAuthV2, self).getConfig()
        config['swift_store_auth_version'] = '2'
        config['swift_store_user'] = 'tenant:user1'
        return config

    def test_v2_with_no_tenant(self):
        uri = 'swift://failme:key@auth_address/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.assertRaises(exceptions.BadStoreUri, self.store.get, loc)

    def test_v2_multi_tenant_location(self):
        config = self.getConfig()
        config['swift_store_multi_tenant'] = True
        self.config(group='swift1', **config)
        uri = 'swift://auth_address/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.assertEqual('swift', loc.store_name)