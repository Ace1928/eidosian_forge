from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareAccess(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareAccess, self).setUp()
        self.shares_mock = self.app.client_manager.share.shares
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)
        self.shares_mock.reset_mock()
        self.access_rules_mock = self.app.client_manager.share.share_access_rules
        self.access_rules_mock.reset_mock()