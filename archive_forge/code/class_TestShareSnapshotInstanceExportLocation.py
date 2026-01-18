from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 import (share_snapshot_instance_export_locations as
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceExportLocation(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareSnapshotInstanceExportLocation, self).setUp()
        self.share_snapshot_instances_mock = self.app.client_manager.share.share_snapshot_instances
        self.share_snapshot_instances_mock.reset_mock()
        self.share_snapshot_instances_el_mock = self.app.client_manager.share.share_snapshot_instance_export_locations
        self.share_snapshot_instances_el_mock.reset_mock()