from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
class VolumesV285Test(VolumesV279Test):
    api_version = '2.85'

    def test_volume_update_server_volume(self):
        v = self.cs.volumes.update_server_volume(server_id=1234, src_volid='Work', dest_volid='Work', delete_on_termination=True)
        self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('PUT', '/servers/1234/os-volume_attachments/Work')
        self.assertIsInstance(v, volumes.Volume)

    def test_volume_update_server_volume_pre_v285(self):
        self.cs.api_version = api_versions.APIVersion('2.84')
        ex = self.assertRaises(TypeError, self.cs.volumes.update_server_volume, '1234', 'Work', 'Work', delete_on_termination=True)
        self.assertIn('delete_on_termination', str(ex))