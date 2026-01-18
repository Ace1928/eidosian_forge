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
class ServersV267Test(ServersV263Test):
    """Tests for creating a server with a block_device_mapping_v2 entry
    using volume_type for microversion 2.67.
    """
    api_version = '2.67'

    def test_create_server_boot_from_volume_with_volume_type(self):
        bdm = [{'volume_size': 1, 'uuid': '11111111-1111-1111-1111-111111111111', 'delete_on_termination': True, 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'volume_type': 'rbd'}]
        s = self.cs.servers.create(name='bfv server', image='', flavor=1, nics='auto', block_device_mapping_v2=bdm)
        self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('POST', '/servers', {'server': {'flavorRef': '1', 'imageRef': '', 'name': 'bfv server', 'networks': 'auto', 'block_device_mapping_v2': bdm, 'min_count': 1, 'max_count': 1}})

    def test_create_server_boot_from_volume_with_volume_type_pre_267(self):
        self.cs.api_version = api_versions.APIVersion('2.66')
        bdm = [{'volume_size': 1, 'uuid': '11111111-1111-1111-1111-111111111111', 'delete_on_termination': True, 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'volume_type': 'rbd'}]
        ex = self.assertRaises(ValueError, self.cs.servers.create, name='bfv server', image='', flavor=1, nics='none', block_device_mapping_v2=bdm)
        self.assertIn('Block device volume_type is not supported before microversion 2.67', str(ex))