from osc_lib import utils as osc_lib_utils
from manilaclient.osc.v2 \
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceExportLocation(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareInstanceExportLocation, self).setUp()
        self.instances_mock = self.app.client_manager.share.share_instances
        self.instances_mock.reset_mock()
        self.instance_export_locations_mock = self.app.client_manager.share.share_instance_export_locations
        self.instance_export_locations_mock.reset_mock()