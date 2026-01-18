from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
class TestTransferList(TestTransfer):
    volume_transfers = volume_fakes.create_one_transfer()

    def setUp(self):
        super().setUp()
        self.transfer_mock.list.return_value = [self.volume_transfers]
        self.cmd = volume_transfer_request.ListTransferRequest(self.app, None)

    def test_transfer_list_without_argument(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['ID', 'Name', 'Volume']
        self.assertEqual(expected_columns, columns)
        datalist = ((self.volume_transfers.id, self.volume_transfers.name, self.volume_transfers.volume_id),)
        self.assertEqual(datalist, tuple(data))
        self.transfer_mock.list.assert_called_with(detailed=True, search_opts={'all_tenants': 0})

    def test_transfer_list_with_argument(self):
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ['ID', 'Name', 'Volume']
        self.assertEqual(expected_columns, columns)
        datalist = ((self.volume_transfers.id, self.volume_transfers.name, self.volume_transfers.volume_id),)
        self.assertEqual(datalist, tuple(data))
        self.transfer_mock.list.assert_called_with(detailed=True, search_opts={'all_tenants': 1})