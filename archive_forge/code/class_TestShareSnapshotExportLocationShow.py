from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotExportLocationShow(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotExportLocationShow, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.export_location = manila_fakes.FakeSnapshotExportLocation.create_one_export_location()
        self.export_locations_mock.get.return_value = self.export_location
        self.cmd = osc_share_snapshots.ShareSnapshotShowExportLocation(self.app, None)

    def test_snapshot_export_locations_list(self):
        arglist = [self.share_snapshot.id, self.export_location.id]
        verifylist = [('snapshot', self.share_snapshot.id), ('export_location', self.export_location.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.export_locations_mock.get.assert_called_with(export_location=self.export_location.id, snapshot=self.share_snapshot)
        self.assertEqual(tuple(self.export_location._info.keys()), columns)
        self.assertCountEqual(self.export_location._info.values(), data)