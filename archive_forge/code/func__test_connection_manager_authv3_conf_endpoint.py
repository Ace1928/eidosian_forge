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
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import backend
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
@mock.patch('keystoneauth1.session.Session.get_endpoint')
@mock.patch('keystoneauth1.session.Session.get_auth_headers', new=mock.Mock())
def _test_connection_manager_authv3_conf_endpoint(self, mock_ep, expected_endpoint='https://from-catalog.com'):
    self.config(swift_store_auth_version='3')
    mock_ep.return_value = 'https://from-catalog.com'
    ctx = mock.MagicMock()
    self.store.configure()
    connection_manager = manager.SingleTenantConnectionManager(store=self.store, store_location=self.location, context=ctx)
    conn = connection_manager._init_connection()
    self.assertEqual(expected_endpoint, conn.preauthurl)