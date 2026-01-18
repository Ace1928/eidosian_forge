from unittest import mock
from openstackclient.identity.v2_0 import catalog
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCatalog(utils.TestCommand):
    service_catalog = identity_fakes.FakeCatalog.create_catalog()

    def setUp(self):
        super(TestCatalog, self).setUp()
        self.sc_mock = mock.Mock()
        self.sc_mock.service_catalog.catalog.return_value = [self.service_catalog]
        self.auth_mock = mock.Mock()
        self.app.client_manager.session = self.auth_mock
        self.auth_mock.auth.get_auth_ref.return_value = self.sc_mock