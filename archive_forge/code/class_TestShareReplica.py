from osc_lib import utils as oscutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareReplica(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareReplica, self).setUp()
        self.replicas_mock = self.app.client_manager.share.share_replicas
        self.replicas_mock.reset_mock()
        self.export_locations_mock = self.app.client_manager.share.share_replica_export_locations
        self.export_locations_mock.reset_mock()