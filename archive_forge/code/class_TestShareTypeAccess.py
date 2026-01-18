from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_type_access as osc_share_type_access
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeAccess(manila_fakes.TestShare):

    def setUp(self):
        super(TestShareTypeAccess, self).setUp()
        self.type_access_mock = self.app.client_manager.share.share_type_access
        self.type_access_mock.reset_mock()
        self.share_types_mock = self.app.client_manager.share.share_types
        self.share_types_mock.reset_mock()
        self.projects_mock = self.app.client_manager.identity.projects
        self.projects_mock.reset_mock()