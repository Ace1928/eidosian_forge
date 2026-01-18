from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
class TestTransferShow(TestTransfer):
    columns = ('created_at', 'id', 'name', 'volume_id')

    def setUp(self):
        super().setUp()
        self.volume_transfer = volume_fakes.create_one_transfer(attrs={'created_at': 'time'})
        self.data = (self.volume_transfer.created_at, self.volume_transfer.id, self.volume_transfer.name, self.volume_transfer.volume_id)
        self.transfer_mock.get.return_value = self.volume_transfer
        self.cmd = volume_transfer_request.ShowTransferRequest(self.app, None)

    def test_transfer_show(self):
        arglist = [self.volume_transfer.id]
        verifylist = [('transfer_request', self.volume_transfer.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.transfer_mock.get.assert_called_once_with(self.volume_transfer.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)