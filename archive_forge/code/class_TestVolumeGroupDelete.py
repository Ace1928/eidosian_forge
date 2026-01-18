from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
class TestVolumeGroupDelete(TestVolumeGroup):
    fake_volume_group = volume_fakes.create_one_volume_group()

    def setUp(self):
        super().setUp()
        self.volume_groups_mock.get.return_value = self.fake_volume_group
        self.volume_groups_mock.delete.return_value = None
        self.cmd = volume_group.DeleteVolumeGroup(self.app, None)

    def test_volume_group_delete(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = [self.fake_volume_group.id, '--force']
        verifylist = [('group', self.fake_volume_group.id), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.delete.assert_called_once_with(self.fake_volume_group.id, delete_volumes=True)
        self.assertIsNone(result)

    def test_volume_group_delete_pre_v313(self):
        self.volume_client.api_version = api_versions.APIVersion('3.12')
        arglist = [self.fake_volume_group.id]
        verifylist = [('group', self.fake_volume_group.id), ('force', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.13 or greater is required', str(exc))