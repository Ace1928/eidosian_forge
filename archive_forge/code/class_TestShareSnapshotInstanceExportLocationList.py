from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 import (share_snapshot_instance_export_locations as
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceExportLocationList(TestShareSnapshotInstanceExportLocation):

    def setUp(self):
        super(TestShareSnapshotInstanceExportLocationList, self).setUp()
        self.share_snapshot_instance = manila_fakes.FakeShareSnapshotIntances.create_one_snapshot_instance()
        self.share_snapshot_instances_export_locations = manila_fakes.FakeShareSnapshotInstancesExportLocations.create_share_snapshot_instances(count=2)
        self.share_snapshot_instances_mock.get.return_value = self.share_snapshot_instance
        self.share_snapshot_instances_el_mock.list.return_value = self.share_snapshot_instances_export_locations
        self.cmd = osc_snapshot_instance_locations.ShareSnapshotInstanceExportLocationList(self.app, None)

    def test_share_snapshot_instance_export_location_list_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_instance_export_location_list(self):
        values = (osc_lib_utils.get_dict_properties(s._info, COLUMNS) for s in self.share_snapshot_instances_export_locations)
        arglist = [self.share_snapshot_instance.id]
        verifylist = [('instance', self.share_snapshot_instance.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(values), list(data))