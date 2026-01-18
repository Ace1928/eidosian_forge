import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
class BaseRackspaceNovaTestCase:
    conn_class = RackspaceNovaMockHttp
    auth_url = 'https://auth.api.example.com'

    def create_driver(self):
        return self.driver_type(*self.driver_args, **self.driver_kwargs)

    def setUp(self):
        self.driver_klass.connectionCls.conn_class = self.conn_class
        self.driver_klass.connectionCls.auth_url = self.auth_url
        self.conn_class.type = None
        self.driver = self.create_driver()
        self.driver.connection._populate_hosts_and_request_paths()
        clear_pricing_data()
        self.node = self.driver.list_nodes()[1]

    def test_service_catalog_contains_right_endpoint(self):
        self.assertEqual(self.driver.connection.get_endpoint(), self.expected_endpoint)

    def test_list_sizes_pricing(self):
        sizes = self.driver.list_sizes()
        for size in sizes:
            if size.ram > 256:
                self.assertTrue(size.price > 0)