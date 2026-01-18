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
class ServersV232Test(ServersV226Test):
    api_version = '2.32'

    def test_create_server_boot_with_tagged_nics(self):
        nics = [{'net-id': '11111111-1111-1111-1111-111111111111', 'tag': 'one'}, {'net-id': '22222222-2222-2222-2222-222222222222', 'tag': 'two'}]
        self.cs.servers.create(name='Server with tagged nics', image=1, flavor=1, nics=nics)
        self.assert_called('POST', '/servers')

    def test_create_server_boot_with_tagged_nics_pre232(self):
        self.cs.api_version = api_versions.APIVersion('2.31')
        nics = [{'net-id': '11111111-1111-1111-1111-111111111111', 'tag': 'one'}, {'net-id': '22222222-2222-2222-2222-222222222222', 'tag': 'two'}]
        self.assertRaises(ValueError, self.cs.servers.create, name='Server with tagged nics', image=1, flavor=1, nics=nics)

    def test_create_server_boot_from_volume_tagged_bdm_v2(self):
        bdm = [{'volume_size': '1', 'volume_id': '11111111-1111-1111-1111-111111111111', 'delete_on_termination': '0', 'device_name': 'vda', 'tag': 'foo'}]
        s = self.cs.servers.create(name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', block_device_mapping_v2=bdm)
        self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers')

    def test_create_server_boot_from_volume_tagged_bdm_v2_pre232(self):
        self.cs.api_version = api_versions.APIVersion('2.31')
        bdm = [{'volume_size': '1', 'volume_id': '11111111-1111-1111-1111-111111111111', 'delete_on_termination': '0', 'device_name': 'vda', 'tag': 'foo'}]
        self.assertRaises(ValueError, self.cs.servers.create, name='My server', image=1, flavor=1, meta={'foo': 'bar'}, userdata='hello moto', key_name='fakekey', block_device_mapping_v2=bdm)