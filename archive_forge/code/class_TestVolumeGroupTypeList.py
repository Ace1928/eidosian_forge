from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
class TestVolumeGroupTypeList(TestVolumeGroupType):
    fake_volume_group_types = volume_fakes.create_volume_group_types()
    columns = ('ID', 'Name', 'Is Public', 'Properties')
    data = [(fake_volume_group_type.id, fake_volume_group_type.name, fake_volume_group_type.is_public, fake_volume_group_type.group_specs) for fake_volume_group_type in fake_volume_group_types]

    def setUp(self):
        super().setUp()
        self.volume_group_types_mock.list.return_value = self.fake_volume_group_types
        self.volume_group_types_mock.default.return_value = self.fake_volume_group_types[0]
        self.cmd = volume_group_type.ListVolumeGroupType(self.app, None)

    def test_volume_group_type_list(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = []
        verifylist = [('show_default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.list.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(tuple(self.data), data)

    def test_volume_group_type_list_with_default_option(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = ['--default']
        verifylist = [('show_default', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.default.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(tuple([self.data[0]]), data)

    def test_volume_group_type_list_pre_v311(self):
        self.volume_client.api_version = api_versions.APIVersion('3.10')
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.11 or greater is required', str(exc))