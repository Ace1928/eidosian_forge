from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTransferDelete(TestShareTransfer):

    def setUp(self):
        super(TestShareTransferDelete, self).setUp()
        self.transfer = manila_fakes.FakeShareTransfer.create_one_transfer()
        self.transfers_mock.get.return_value = self.transfer
        self.cmd = osc_share_transfers.DeleteShareTransfer(self.app, None)

    def test_share_transfer_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_transfer_delete(self):
        arglist = [self.transfer.id]
        verifylist = [('transfer', [self.transfer.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.transfers_mock.delete.assert_called_with(self.transfer.id)
        self.assertIsNone(result)

    def test_share_transfer_delete_multiple(self):
        transfers = manila_fakes.FakeShareTransfer.create_share_transfers(count=2)
        arglist = [transfers[0].id, transfers[1].id]
        verifylist = [('transfer', [transfers[0].id, transfers[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.transfers_mock.delete.call_count, len(transfers))
        self.assertIsNone(result)

    def test_share_transfer_delete_exception(self):
        arglist = [self.transfer.id]
        verifylist = [('transfer', [self.transfer.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.transfers_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)