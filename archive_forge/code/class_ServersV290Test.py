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
class ServersV290Test(ServersV278Test):
    api_version = '2.90'

    def test_create_server_with_hostname(self):
        self.cs.servers.create(name='My server', image=1, flavor=1, nics='auto', hostname='new-hostname')
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '1', 'max_count': 1, 'min_count': 1, 'name': 'My server', 'networks': 'auto', 'hostname': 'new-hostname'}})

    def test_create_server_with_hostname_pre_290_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.89')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.create, name='My server', image=1, flavor=1, nics='auto', hostname='new-hostname')
        self.assertIn("'hostname' argument is only allowed since microversion 2.90", str(ex))

    def test_rebuild_server_with_hostname(self):
        s = self.cs.servers.get(1234)
        ret = s.rebuild(image='1', hostname='new-hostname')
        self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers/1234/action', {'rebuild': {'imageRef': '1', 'hostname': 'new-hostname'}})

    def test_rebuild_server_with_hostname_pre_290_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.89')
        ex = self.assertRaises(exceptions.UnsupportedAttribute, self.cs.servers.rebuild, '1234', fakes.FAKE_IMAGE_UUID_1, hostname='new-hostname')
        self.assertIn('hostname', str(ex))

    def test_update_server_with_hostname(self):
        s = self.cs.servers.get(1234)
        s.update(hostname='new-hostname')
        self.assert_called('PUT', '/servers/1234', {'server': {'hostname': 'new-hostname'}})

    def test_update_with_hostname_pre_290_fails(self):
        self.cs.api_version = api_versions.APIVersion('2.89')
        s = self.cs.servers.get(1234)
        ex = self.assertRaises(TypeError, s.update, hostname='new-hostname')
        self.assertIn('hostname', str(ex))