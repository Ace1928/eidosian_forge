from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as osc_lib_utils
from manilaclient.common.apiclient import exceptions as api_exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstance(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareSnapshotInstance, self).setUp()
        self.share_snapshots_mock = self.app.client_manager.share.share_snapshots
        self.share_snapshots_mock.reset_mock()
        self.share_snapshot_instances_mock = self.app.client_manager.share.share_snapshot_instances
        self.share_snapshot_instances_mock.reset_mock()
        self.share_snapshot_instances_el_mock = self.app.client_manager.share.share_snapshot_instance_export_locations
        self.share_snapshot_instances_el_mock.reset_mock()