import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
class ServersV274Test(ServersV273Test):
    api_version = '2.74'

    def test_create_server_with_host(self):
        self.cs.servers.create(name='My server', image=1, flavor=1, nics='auto', host='new-host')
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'max_count': 1, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'host': 'new-host'}})

    def test_create_server_with_hypervisor_hostname(self):
        self.cs.servers.create(name='My server', image=1, flavor=1, nics='auto', hypervisor_hostname='new-host')
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'max_count': 1, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'hypervisor_hostname': 'new-host'}})

    def test_create_server_with_host_and_hypervisor_hostname(self):
        self.cs.servers.create(name='My server', image=1, flavor=1, nics='auto', host='new-host', hypervisor_hostname='new-host')
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'max_count': 1, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'host': 'new-host', 'hypervisor_hostname': 'new-host'}})

    def test_create_server_with_host_pre_274_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.73')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, nics='auto', host='new-host')
        self.assertIn("'host' argument is only allowed since microversion 2.74", str(ex))

    def test_create_server_with_hypervisor_hostname_pre_274_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.73')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, nics='auto', hypervisor_hostname='new-host')
        self.assertIn("'hypervisor_hostname' argument is only allowed since microversion 2.74", str(ex))

    def test_create_server_with_host_and_hypervisor_hostname_pre_274_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.73')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, nics='auto', host='new-host', hypervisor_hostname='new-host')
        self.assertIn("'host' argument is only allowed since microversion 2.74", str(ex))