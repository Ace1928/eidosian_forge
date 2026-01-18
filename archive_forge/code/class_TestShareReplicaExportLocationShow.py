from osc_lib import utils as oscutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplicaExportLocationShow(TestShareReplica):

    def setUp(self):
        super(TestShareReplicaExportLocationShow, self).setUp()
        self.share_replica = manila_fakes.FakeShareReplica.create_one_replica()
        self.replicas_mock.get.return_value = self.share_replica
        self.export_location = manila_fakes.FakeShareExportLocation.create_one_export_location()
        self.export_locations_mock.get.return_value = self.export_location
        self.cmd = osc_replica_el.ShareReplicaShowExportLocation(self.app, None)

    def test_replica_export_locations_show(self):
        arglist = [self.share_replica.id, self.export_location.id]
        verifylist = [('replica', self.share_replica.id), ('export_location', self.export_location.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.get.assert_called_with(self.share_replica.id)
        self.export_locations_mock.get.assert_called_with(self.share_replica, self.export_location.id)
        self.assertCountEqual(tuple(self.export_location._info.keys()), columns)
        self.assertCountEqual(self.export_location._info.values(), data)