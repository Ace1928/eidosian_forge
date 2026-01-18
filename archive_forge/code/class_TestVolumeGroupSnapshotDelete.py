from unittest import mock
from keystoneauth1 import discover
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.test import fakes as sdk_fakes
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_snapshot
class TestVolumeGroupSnapshotDelete(volume_fakes.TestVolume):
    fake_volume_group_snapshot = sdk_fakes.generate_fake_resource(_group_snapshot.GroupSnapshot)

    def setUp(self):
        super().setUp()
        self.volume_sdk_client.find_group_snapshot.return_value = self.fake_volume_group_snapshot
        self.volume_sdk_client.delete_group_snapshot.return_value = None
        self.cmd = volume_group_snapshot.DeleteVolumeGroupSnapshot(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_volume_group_snapshot_delete(self, mock_mv):
        mock_mv.side_effect = fake_supports_microversion('3.14')
        arglist = [self.fake_volume_group_snapshot.id]
        verifylist = [('snapshot', self.fake_volume_group_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.delete_group_snapshot.assert_called_once_with(self.fake_volume_group_snapshot.id)
        self.assertIsNone(result)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_volume_group_snapshot_delete_pre_v314(self, mock_mv):
        mock_mv.side_effect = fake_supports_microversion('3.13')
        arglist = [self.fake_volume_group_snapshot.id]
        verifylist = [('snapshot', self.fake_volume_group_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.14 or greater is required', str(exc))