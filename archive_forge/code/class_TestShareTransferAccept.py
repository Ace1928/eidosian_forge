from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTransferAccept(TestShareTransfer):

    def setUp(self):
        super(TestShareTransferAccept, self).setUp()
        self.transfer = manila_fakes.FakeShareTransfer.create_one_transfer()
        self.transfers_mock.get.return_value = self.transfer
        self.cmd = osc_share_transfers.AcceptShareTransfer(self.app, None)

    def test_share_transfer_accept_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_transfer_accept(self):
        arglist = [self.transfer.id, self.transfer.auth_key]
        verifylist = [('transfer', self.transfer.id), ('auth_key', self.transfer.auth_key)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.transfers_mock.accept.assert_called_with(self.transfer.id, self.transfer.auth_key, clear_access_rules=False)
        self.assertIsNone(result)