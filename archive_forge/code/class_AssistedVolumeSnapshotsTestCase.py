from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class AssistedVolumeSnapshotsTestCase(utils.TestCase):

    def setUp(self):
        super(AssistedVolumeSnapshotsTestCase, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.1'))

    def test_create_snap(self):
        vs = self.cs.assisted_volume_snapshots.create('1', {})
        self.assert_request_id(vs, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('POST', '/os-assisted-volume-snapshots')

    def test_delete_snap(self):
        vs = self.cs.assisted_volume_snapshots.delete('x', {})
        self.assert_request_id(vs, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('DELETE', '/os-assisted-volume-snapshots/x?delete_info={}')