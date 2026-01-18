import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSecurityService(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareSecurityService, self).setUp()
        self.security_services_mock = self.app.client_manager.share.security_services
        self.security_services_mock.reset_mock()
        self.share_networks_mock = self.app.client_manager.share.share_networks
        self.share_networks_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)