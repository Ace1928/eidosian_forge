from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroupAddVolume(TestConsistencyGroup):
    _consistency_group = volume_fakes.create_one_consistency_group()

    def setUp(self):
        super().setUp()
        self.consistencygroups_mock.get.return_value = self._consistency_group
        self.cmd = consistency_group.AddVolumeToConsistencyGroup(self.app, None)

    def test_add_one_volume_to_consistency_group(self):
        volume = volume_fakes.create_one_volume()
        self.volumes_mock.get.return_value = volume
        arglist = [self._consistency_group.id, volume.id]
        verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volume.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'add_volumes': volume.id}
        self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, **kwargs)
        self.assertIsNone(result)

    def test_add_multiple_volumes_to_consistency_group(self):
        volumes = volume_fakes.create_volumes(count=2)
        self.volumes_mock.get = volume_fakes.get_volumes(volumes)
        arglist = [self._consistency_group.id, volumes[0].id, volumes[1].id]
        verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volumes[0].id, volumes[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'add_volumes': volumes[0].id + ',' + volumes[1].id}
        self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, **kwargs)
        self.assertIsNone(result)

    @mock.patch.object(consistency_group.LOG, 'error')
    def test_add_multiple_volumes_to_consistency_group_with_exception(self, mock_error):
        volume = volume_fakes.create_one_volume()
        arglist = [self._consistency_group.id, volume.id, 'unexist_volume']
        verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volume.id, 'unexist_volume'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [volume, exceptions.CommandError, self._consistency_group]
        with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
            result = self.cmd.take_action(parsed_args)
            mock_error.assert_called_with('1 of 2 volumes failed to add.')
            self.assertIsNone(result)
            find_mock.assert_any_call(self.consistencygroups_mock, self._consistency_group.id)
            find_mock.assert_any_call(self.volumes_mock, volume.id)
            find_mock.assert_any_call(self.volumes_mock, 'unexist_volume')
            self.assertEqual(3, find_mock.call_count)
            self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, add_volumes=volume.id)