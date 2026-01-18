from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as osc_lib_utils
from manilaclient.common.apiclient import exceptions as api_exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceList(TestShareSnapshotInstance):

    def setUp(self):
        super(TestShareSnapshotInstanceList, self).setUp()
        self.share_snapshot_instances = manila_fakes.FakeShareSnapshotIntances.create_share_snapshot_instances(count=2)
        self.share_snapshot_instances_mock.list.return_value = self.share_snapshot_instances
        self.cmd = osc_share_snapshot_instances.ListShareSnapshotInstance(self.app, None)

    def test_share_snapshot_instance_list(self):
        values = (osc_lib_utils.get_dict_properties(s._info, COLUMNS) for s in self.share_snapshot_instances)
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(values), list(data))

    def test_share_snapshot_instance_list_detail(self):
        values = (osc_lib_utils.get_dict_properties(s._info, COLUMNS_DETAIL) for s in self.share_snapshot_instances)
        arglist = ['--detailed']
        verifylist = [('detailed', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(COLUMNS_DETAIL, columns)
        self.assertEqual(list(values), list(data))

    def test_share_snapshot_instance_list_snapshot_id(self):
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.share_snapshots_mock.get.return_value = self.share_snapshot
        self.share_snapshot_instances_mock.list.return_value = [self.share_snapshot_instances[0]]
        values = (osc_lib_utils.get_dict_properties(s._info, COLUMNS) for s in [self.share_snapshot_instances[0]])
        arglist = ['--snapshot', self.share_snapshot.id]
        verifylist = [('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(values), list(data))