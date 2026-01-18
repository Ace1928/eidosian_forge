from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstance(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareInstance, self).setUp()
        self.shares_mock = self.app.client_manager.share.shares
        self.shares_mock.reset_mock()
        self.instances_mock = self.app.client_manager.share.share_instances
        self.instances_mock.reset_mock()
        self.share_instance_export_locations_mock = self.app.client_manager.share.share_instance_export_locations
        self.share_instance_export_locations_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)