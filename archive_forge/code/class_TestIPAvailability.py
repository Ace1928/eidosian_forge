from unittest import mock
from osc_lib.cli import format_columns
from openstackclient.network.v2 import ip_availability
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestIPAvailability(network_fakes.TestNetworkV2):

    def setUp(self):
        super(TestIPAvailability, self).setUp()
        self.projects_mock = self.app.client_manager.identity.projects
        self.project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = self.project