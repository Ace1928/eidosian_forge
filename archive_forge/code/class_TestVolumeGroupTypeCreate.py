from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
class TestVolumeGroupTypeCreate(TestVolumeGroupType):
    maxDiff = 2000
    fake_volume_group_type = volume_fakes.create_one_volume_group_type()
    columns = ('ID', 'Name', 'Description', 'Is Public', 'Properties')
    data = (fake_volume_group_type.id, fake_volume_group_type.name, fake_volume_group_type.description, fake_volume_group_type.is_public, format_columns.DictColumn(fake_volume_group_type.group_specs))

    def setUp(self):
        super().setUp()
        self.volume_group_types_mock.create.return_value = self.fake_volume_group_type
        self.cmd = volume_group_type.CreateVolumeGroupType(self.app, None)

    def test_volume_group_type_create(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = [self.fake_volume_group_type.name]
        verifylist = [('name', self.fake_volume_group_type.name), ('description', None), ('is_public', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.create.assert_called_once_with(self.fake_volume_group_type.name, None, True)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_type_create_with_options(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = [self.fake_volume_group_type.name, '--description', 'foo', '--private']
        verifylist = [('name', self.fake_volume_group_type.name), ('description', 'foo'), ('is_public', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.create.assert_called_once_with(self.fake_volume_group_type.name, 'foo', False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_type_create_pre_v311(self):
        self.volume_client.api_version = api_versions.APIVersion('3.10')
        arglist = [self.fake_volume_group_type.name]
        verifylist = [('name', self.fake_volume_group_type.name), ('description', None), ('is_public', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.11 or greater is required', str(exc))