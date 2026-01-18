from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as osc_lib_utils
from manilaclient.common.apiclient import exceptions as api_exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceShow(TestShareSnapshotInstance):

    def setUp(self):
        super(TestShareSnapshotInstanceShow, self).setUp()
        self.share_snapshot_instance = manila_fakes.FakeShareSnapshotIntances.create_one_snapshot_instance()
        self.share_snapshot_instances_mock.get.return_value = self.share_snapshot_instance
        self.share_snapshot_instances_el_list = manila_fakes.FakeShareSnapshotInstancesExportLocations.create_share_snapshot_instances(count=2)
        self.share_snapshot_instances_el_mock.list.return_value = self.share_snapshot_instances_el_list
        self.cmd = osc_share_snapshot_instances.ShowShareSnapshotInstance(self.app, None)
        self.share_snapshot_instance._info['export_locations'] = cliutils.convert_dict_list_to_string(self.share_snapshot_instances_el_list)
        self.data = self.share_snapshot_instance._info.values()
        self.columns = self.share_snapshot_instance._info.keys()

    def test_share_snapshot_instance_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_instance_show(self):
        arglist = [self.share_snapshot_instance.id]
        verifylist = [('snapshot_instance', self.share_snapshot_instance.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_snapshot_instances_mock.get.assert_called_with(self.share_snapshot_instance.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)