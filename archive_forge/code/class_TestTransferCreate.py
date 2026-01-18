from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
class TestTransferCreate(TestTransfer):
    volume = volume_fakes.create_one_volume()
    columns = ('auth_key', 'created_at', 'id', 'name', 'volume_id')

    def setUp(self):
        super().setUp()
        self.volume_transfer = volume_fakes.create_one_transfer(attrs={'volume_id': self.volume.id, 'auth_key': 'key', 'created_at': 'time'})
        self.data = (self.volume_transfer.auth_key, self.volume_transfer.created_at, self.volume_transfer.id, self.volume_transfer.name, self.volume_transfer.volume_id)
        self.transfer_mock.create.return_value = self.volume_transfer
        self.volumes_mock.get.return_value = self.volume
        self.cmd = volume_transfer_request.CreateTransferRequest(self.app, None)

    def test_transfer_create_without_name(self):
        arglist = [self.volume.id]
        verifylist = [('volume', self.volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfer_mock.create.assert_called_once_with(self.volume.id, None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_transfer_create_with_name(self):
        arglist = ['--name', self.volume_transfer.name, self.volume.id]
        verifylist = [('name', self.volume_transfer.name), ('volume', self.volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfer_mock.create.assert_called_once_with(self.volume.id, self.volume_transfer.name)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_transfer_create_with_no_snapshots(self):
        self.volume_client.api_version = api_versions.APIVersion('3.55')
        arglist = ['--no-snapshots', self.volume.id]
        verifylist = [('name', None), ('snapshots', False), ('volume', self.volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfer_mock.create.assert_called_once_with(self.volume.id, None, no_snapshots=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_transfer_create_pre_v355(self):
        self.volume_client.api_version = api_versions.APIVersion('3.54')
        arglist = ['--no-snapshots', self.volume.id]
        verifylist = [('name', None), ('snapshots', False), ('volume', self.volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.55 or greater is required', str(exc))