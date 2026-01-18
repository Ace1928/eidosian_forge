from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
class TestTypeList(TestType):
    volume_types = volume_fakes.create_volume_types()
    columns = ['ID', 'Name', 'Is Public']
    columns_long = columns + ['Description', 'Properties']
    data_with_default_type = [(volume_types[0].id, volume_types[0].name, True)]
    data = []
    for t in volume_types:
        data.append((t.id, t.name, t.is_public))
    data_long = []
    for t in volume_types:
        data_long.append((t.id, t.name, t.is_public, t.description, format_columns.DictColumn(t.extra_specs)))

    def setUp(self):
        super().setUp()
        self.volume_types_mock.list.return_value = self.volume_types
        self.volume_types_mock.default.return_value = self.volume_types[0]
        self.cmd = volume_type.ListVolumeType(self.app, None)

    def test_type_list_without_options(self):
        arglist = []
        verifylist = [('long', False), ('is_public', None), ('default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.list.assert_called_once_with(search_opts={}, is_public=None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_type_list_with_options(self):
        arglist = ['--long', '--public']
        verifylist = [('long', True), ('is_public', True), ('default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.list.assert_called_once_with(search_opts={}, is_public=True)
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_type_list_with_private_option(self):
        arglist = ['--private']
        verifylist = [('long', False), ('is_public', False), ('default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.list.assert_called_once_with(search_opts={}, is_public=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_type_list_with_default_option(self):
        arglist = ['--default']
        verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.default.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data_with_default_type, list(data))

    def test_type_list_with_properties(self):
        self.app.client_manager.volume.api_version = api_versions.APIVersion('3.52')
        arglist = ['--property', 'foo=bar', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1']
        verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', False), ('properties', {'foo': 'bar'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.list.assert_called_once_with(search_opts={'extra_specs': {'foo': 'bar', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'}}, is_public=None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_type_list_with_properties_pre_v352(self):
        self.app.client_manager.volume.api_version = api_versions.APIVersion('3.51')
        arglist = ['--property', 'foo=bar']
        verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', False), ('properties', {'foo': 'bar'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.52 or greater is required', str(exc))

    def test_type_list_with_encryption(self):
        encryption_type = volume_fakes.create_one_encryption_volume_type(attrs={'volume_type_id': self.volume_types[0].id})
        encryption_info = {'provider': 'LuksEncryptor', 'cipher': None, 'key_size': None, 'control_location': 'front-end'}
        encryption_columns = self.columns + ['Encryption']
        encryption_data = []
        encryption_data.append((self.volume_types[0].id, self.volume_types[0].name, self.volume_types[0].is_public, volume_type.EncryptionInfoColumn(self.volume_types[0].id, {self.volume_types[0].id: encryption_info})))
        encryption_data.append((self.volume_types[1].id, self.volume_types[1].name, self.volume_types[1].is_public, volume_type.EncryptionInfoColumn(self.volume_types[1].id, {})))
        self.volume_encryption_types_mock.list.return_value = [encryption_type]
        arglist = ['--encryption-type']
        verifylist = [('encryption_type', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_encryption_types_mock.list.assert_called_once_with()
        self.volume_types_mock.list.assert_called_once_with(search_opts={}, is_public=None)
        self.assertEqual(encryption_columns, columns)
        self.assertCountEqual(encryption_data, list(data))