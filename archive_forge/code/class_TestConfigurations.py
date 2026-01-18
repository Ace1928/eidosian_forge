from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurations(fakes.TestDatabasev1):
    fake_configurations = fakes.FakeConfigurations()
    fake_configuration_params = fakes.FakeConfigurationParameters()

    def setUp(self):
        super(TestConfigurations, self).setUp()
        self.mock_client = self.app.client_manager.database
        self.configuration_client = self.app.client_manager.database.configurations
        self.instance_client = self.app.client_manager.database.instances
        self.configuration_params_client = self.app.client_manager.database.configuration_parameters