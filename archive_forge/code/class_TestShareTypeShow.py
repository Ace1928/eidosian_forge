from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeShow(TestShareType):

    def setUp(self):
        super(TestShareTypeShow, self).setUp()
        self.share_type = manila_fakes.FakeShareType.create_one_sharetype()
        self.shares_mock.get.return_value = self.share_type
        self.cmd = osc_share_types.ShowShareType(self.app, None)
        self.data = [self.share_type.id, self.share_type.name, 'public', self.share_type.is_default, 'driver_handles_share_servers : True', 'replication_type : readable\nmount_snapshot_support : False\nrevert_to_snapshot_support : False\ncreate_share_from_snapshot_support : True\nsnapshot_support : True', self.share_type.description]
        self.raw_data = [self.share_type.id, self.share_type.name, 'public', self.share_type.is_default, {'driver_handles_share_servers': True}, {'replication_type': 'readable', 'mount_snapshot_support': False, 'revert_to_snapshot_support': False, 'create_share_from_snapshot_support': True, 'snapshot_support': True}, self.share_type.description]

    def test_share_type_show(self):
        arglist = [self.share_type.id]
        verifylist = [('share_type', self.share_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share_type.id)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.data, data)

    def test_share_type_show_json_format(self):
        arglist = [self.share_type.id, '-f', 'json']
        verifylist = [('share_type', self.share_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share_type.id)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.raw_data, data)