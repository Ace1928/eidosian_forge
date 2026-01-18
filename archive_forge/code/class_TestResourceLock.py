from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLock(manila_fakes.TestShare):

    def setUp(self):
        super(TestResourceLock, self).setUp()
        self.shares_mock = self.app.client_manager.share.shares
        self.shares_mock.reset_mock()
        self.locks_mock = self.app.client_manager.share.resource_locks
        self.locks_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)