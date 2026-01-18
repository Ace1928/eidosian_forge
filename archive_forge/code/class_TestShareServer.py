from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareServer(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareServer, self).setUp()
        self.servers_mock = self.app.client_manager.share.share_servers
        self.servers_mock.reset_mock()
        self.share_networks_mock = self.app.client_manager.share.share_networks
        self.share_networks_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)