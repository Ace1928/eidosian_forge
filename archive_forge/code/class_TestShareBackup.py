from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareBackup(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareBackup, self).setUp()
        self.shares_mock = self.app.client_manager.share.shares
        self.shares_mock.reset_mock()
        self.backups_mock = self.app.client_manager.share.share_backups
        self.backups_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(MAX_VERSION)