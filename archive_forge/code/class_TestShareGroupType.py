from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupType(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareGroupType, self).setUp()
        self.sgt_mock = self.app.client_manager.share.share_group_types
        self.sgt_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(api_versions.MAX_VERSION)