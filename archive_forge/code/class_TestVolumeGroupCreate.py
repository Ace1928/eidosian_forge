from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
class TestVolumeGroupCreate(TestVolumeGroup):
    fake_volume_type = volume_fakes.create_one_volume_type()
    fake_volume_group_type = volume_fakes.create_one_volume_group_type()
    fake_volume_group = volume_fakes.create_one_volume_group(attrs={'group_type': fake_volume_group_type.id, 'volume_types': [fake_volume_type.id]})
    fake_volume_group_snapshot = volume_fakes.create_one_volume_group_snapshot()
    columns = ('ID', 'Status', 'Name', 'Description', 'Group Type', 'Volume Types', 'Availability Zone', 'Created At', 'Volumes', 'Group Snapshot ID', 'Source Group ID')
    data = (fake_volume_group.id, fake_volume_group.status, fake_volume_group.name, fake_volume_group.description, fake_volume_group.group_type, fake_volume_group.volume_types, fake_volume_group.availability_zone, fake_volume_group.created_at, fake_volume_group.volumes, fake_volume_group.group_snapshot_id, fake_volume_group.source_group_id)

    def setUp(self):
        super().setUp()
        self.volume_types_mock.get.return_value = self.fake_volume_type
        self.volume_group_types_mock.get.return_value = self.fake_volume_group_type
        self.volume_groups_mock.create.return_value = self.fake_volume_group
        self.volume_groups_mock.get.return_value = self.fake_volume_group
        self.volume_groups_mock.create_from_src.return_value = self.fake_volume_group
        self.volume_group_snapshots_mock.get.return_value = self.fake_volume_group_snapshot
        self.cmd = volume_group.CreateVolumeGroup(self.app, None)

    def test_volume_group_create(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = ['--volume-group-type', self.fake_volume_group_type.id, '--volume-type', self.fake_volume_type.id]
        verifylist = [('volume_group_type', self.fake_volume_group_type.id), ('volume_types', [self.fake_volume_type.id]), ('name', None), ('description', None), ('availability_zone', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
        self.volume_types_mock.get.assert_called_once_with(self.fake_volume_type.id)
        self.volume_groups_mock.create.assert_called_once_with(self.fake_volume_group_type.id, self.fake_volume_type.id, None, None, availability_zone=None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_create__legacy(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = [self.fake_volume_group_type.id, self.fake_volume_type.id]
        verifylist = [('volume_group_type_legacy', self.fake_volume_group_type.id), ('volume_types_legacy', [self.fake_volume_type.id]), ('name', None), ('description', None), ('availability_zone', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
            columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
        self.volume_types_mock.get.assert_called_once_with(self.fake_volume_type.id)
        self.volume_groups_mock.create.assert_called_once_with(self.fake_volume_group_type.id, self.fake_volume_type.id, None, None, availability_zone=None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)
        mock_warning.assert_called_once()
        self.assertIn('Passing volume group type and volume types as positional ', str(mock_warning.call_args[0][0]))

    def test_volume_group_create_no_volume_type(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = ['--volume-group-type', self.fake_volume_group_type.id]
        verifylist = [('volume_group_type', self.fake_volume_group_type.id), ('name', None), ('description', None), ('availability_zone', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--volume-types is a required argument when creating ', str(exc))

    def test_volume_group_create_with_options(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = ['--volume-group-type', self.fake_volume_group_type.id, '--volume-type', self.fake_volume_type.id, '--name', 'foo', '--description', 'hello, world', '--availability-zone', 'bar']
        verifylist = [('volume_group_type', self.fake_volume_group_type.id), ('volume_types', [self.fake_volume_type.id]), ('name', 'foo'), ('description', 'hello, world'), ('availability_zone', 'bar')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
        self.volume_types_mock.get.assert_called_once_with(self.fake_volume_type.id)
        self.volume_groups_mock.create.assert_called_once_with(self.fake_volume_group_type.id, self.fake_volume_type.id, 'foo', 'hello, world', availability_zone='bar')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_create_pre_v313(self):
        self.volume_client.api_version = api_versions.APIVersion('3.12')
        arglist = ['--volume-group-type', self.fake_volume_group_type.id, '--volume-type', self.fake_volume_type.id]
        verifylist = [('volume_group_type', self.fake_volume_group_type.id), ('volume_types', [self.fake_volume_type.id]), ('name', None), ('description', None), ('availability_zone', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.13 or greater is required', str(exc))

    def test_volume_group_create_from_source_group(self):
        self.volume_client.api_version = api_versions.APIVersion('3.14')
        arglist = ['--source-group', self.fake_volume_group.id]
        verifylist = [('source_group', self.fake_volume_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.get.assert_has_calls([mock.call(self.fake_volume_group.id), mock.call(self.fake_volume_group.id)])
        self.volume_groups_mock.create_from_src.assert_called_once_with(None, self.fake_volume_group.id, None, None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_create_from_group_snapshot(self):
        self.volume_client.api_version = api_versions.APIVersion('3.14')
        arglist = ['--group-snapshot', self.fake_volume_group_snapshot.id]
        verifylist = [('group_snapshot', self.fake_volume_group_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_snapshots_mock.get.assert_called_once_with(self.fake_volume_group_snapshot.id)
        self.volume_groups_mock.get.assert_called_once_with(self.fake_volume_group.id)
        self.volume_groups_mock.create_from_src.assert_called_once_with(self.fake_volume_group_snapshot.id, None, None, None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_create_from_src_pre_v314(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = ['--source-group', self.fake_volume_group.id]
        verifylist = [('source_group', self.fake_volume_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.14 or greater is required', str(exc))

    def test_volume_group_create_from_src_source_group_group_snapshot(self):
        self.volume_client.api_version = api_versions.APIVersion('3.14')
        arglist = ['--source-group', self.fake_volume_group.id, '--group-snapshot', self.fake_volume_group_snapshot.id]
        verifylist = [('source_group', self.fake_volume_group.id), ('group_snapshot', self.fake_volume_group_snapshot.id)]
        exc = self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertIn('--group-snapshot: not allowed with argument --source-group', str(exc))