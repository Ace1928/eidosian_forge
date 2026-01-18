import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
class OpenStackServiceCatalogTestCase(unittest.TestCase):
    fixtures = ComputeFileFixtures('openstack')

    def test_parsing_auth_v1_1(self):
        data = self.fixtures.load('_v1_1__auth.json')
        data = json.loads(data)
        service_catalog = data['auth']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='1.0')
        entries = catalog.get_entries()
        self.assertEqual(len(entries), 3)
        entry = [e for e in entries if e.service_type == 'cloudFilesCDN'][0]
        self.assertEqual(entry.service_type, 'cloudFilesCDN')
        self.assertIsNone(entry.service_name)
        self.assertEqual(len(entry.endpoints), 2)
        self.assertEqual(entry.endpoints[0].region, 'ORD')
        self.assertEqual(entry.endpoints[0].url, 'https://cdn2.clouddrive.com/v1/MossoCloudFS')
        self.assertEqual(entry.endpoints[0].endpoint_type, 'external')
        self.assertEqual(entry.endpoints[1].region, 'LON')
        self.assertEqual(entry.endpoints[1].endpoint_type, 'external')

    def test_parsing_auth_v2(self):
        data = self.fixtures.load('_v2_0__auth.json')
        data = json.loads(data)
        service_catalog = data['access']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
        entries = catalog.get_entries()
        self.assertEqual(len(entries), 10)
        entry = [e for e in entries if e.service_name == 'cloudServers'][0]
        self.assertEqual(entry.service_type, 'compute')
        self.assertEqual(entry.service_name, 'cloudServers')
        self.assertEqual(len(entry.endpoints), 1)
        self.assertIsNone(entry.endpoints[0].region)
        self.assertEqual(entry.endpoints[0].url, 'https://servers.api.rackspacecloud.com/v1.0/1337')
        self.assertEqual(entry.endpoints[0].endpoint_type, 'external')

    def test_parsing_auth_v3(self):
        data = self.fixtures.load('_v3__auth.json')
        data = json.loads(data)
        service_catalog = data['token']['catalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='3.x')
        entries = catalog.get_entries()
        self.assertEqual(len(entries), 6)
        entry = [e for e in entries if e.service_type == 'volume'][0]
        self.assertEqual(entry.service_type, 'volume')
        self.assertIsNone(entry.service_name)
        self.assertEqual(len(entry.endpoints), 3)
        self.assertEqual(entry.endpoints[0].region, 'regionOne')
        self.assertEqual(entry.endpoints[0].endpoint_type, 'external')
        self.assertEqual(entry.endpoints[1].region, 'regionOne')
        self.assertEqual(entry.endpoints[1].endpoint_type, 'admin')
        self.assertEqual(entry.endpoints[2].region, 'regionOne')
        self.assertEqual(entry.endpoints[2].endpoint_type, 'internal')

    def test_get_public_urls(self):
        data = self.fixtures.load('_v2_0__auth.json')
        data = json.loads(data)
        service_catalog = data['access']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
        public_urls = catalog.get_public_urls(service_type='object-store')
        expected_urls = ['https://storage101.lon1.clouddrive.com/v1/MossoCloudFS_11111-111111111-1111111111-1111111', 'https://storage101.ord1.clouddrive.com/v1/MossoCloudFS_11111-111111111-1111111111-1111111']
        self.assertEqual(public_urls, expected_urls)

    def test_get_regions(self):
        data = self.fixtures.load('_v2_0__auth.json')
        data = json.loads(data)
        service_catalog = data['access']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
        regions = catalog.get_regions(service_type='object-store')
        self.assertEqual(regions, ['LON', 'ORD'])
        regions = catalog.get_regions(service_type='invalid')
        self.assertEqual(regions, [])

    def test_get_service_types(self):
        data = self.fixtures.load('_v2_0__auth.json')
        data = json.loads(data)
        service_catalog = data['access']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
        service_types = catalog.get_service_types()
        self.assertEqual(service_types, ['compute', 'image', 'network', 'object-store', 'rax:object-cdn', 'volumev2', 'volumev3'])
        service_types = catalog.get_service_types(region='ORD')
        self.assertEqual(service_types, ['rax:object-cdn'])

    def test_get_service_names(self):
        data = self.fixtures.load('_v2_0__auth.json')
        data = json.loads(data)
        service_catalog = data['access']['serviceCatalog']
        catalog = OpenStackServiceCatalog(service_catalog=service_catalog, auth_version='2.0')
        service_names = catalog.get_service_names()
        self.assertEqual(service_names, ['cinderv2', 'cinderv3', 'cloudFiles', 'cloudFilesCDN', 'cloudServers', 'cloudServersOpenStack', 'cloudServersPreprod', 'glance', 'neutron', 'nova'])
        service_names = catalog.get_service_names(service_type='compute')
        self.assertEqual(service_names, ['cloudServers', 'cloudServersOpenStack', 'cloudServersPreprod', 'nova'])