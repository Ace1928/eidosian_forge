import copy
from openstackclient.identity.v3 import unscoped_saml
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestUnscopedSAML(identity_fakes.TestFederatedIdentity):

    def setUp(self):
        super(TestUnscopedSAML, self).setUp()
        federation_lib = self.app.client_manager.identity.federation
        self.projects_mock = federation_lib.projects
        self.projects_mock.reset_mock()
        self.domains_mock = federation_lib.domains
        self.domains_mock.reset_mock()