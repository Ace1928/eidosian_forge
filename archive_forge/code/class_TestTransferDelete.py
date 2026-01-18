from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
class TestTransferDelete(TestTransfer):
    volume_transfers = volume_fakes.create_transfers(count=2)

    def setUp(self):
        super().setUp()
        self.transfer_mock.get = volume_fakes.get_transfers(self.volume_transfers)
        self.transfer_mock.delete.return_value = None
        self.cmd = volume_transfer_request.DeleteTransferRequest(self.app, None)

    def test_transfer_delete(self):
        arglist = [self.volume_transfers[0].id]
        verifylist = [('transfer_request', [self.volume_transfers[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.transfer_mock.delete.assert_called_with(self.volume_transfers[0].id)
        self.assertIsNone(result)

    def test_delete_multiple_transfers(self):
        arglist = []
        for v in self.volume_transfers:
            arglist.append(v.id)
        verifylist = [('transfer_request', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for v in self.volume_transfers:
            calls.append(call(v.id))
        self.transfer_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_transfers_with_exception(self):
        arglist = [self.volume_transfers[0].id, 'unexist_transfer']
        verifylist = [('transfer_request', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self.volume_transfers[0], exceptions.CommandError]
        with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
            try:
                self.cmd.take_action(parsed_args)
                self.fail('CommandError should be raised.')
            except exceptions.CommandError as e:
                self.assertEqual('1 of 2 volume transfer requests failed to delete', str(e))
            find_mock.assert_any_call(self.transfer_mock, self.volume_transfers[0].id)
            find_mock.assert_any_call(self.transfer_mock, 'unexist_transfer')
            self.assertEqual(2, find_mock.call_count)
            self.transfer_mock.delete.assert_called_once_with(self.volume_transfers[0].id)