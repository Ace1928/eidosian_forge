import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
class TestDeployment(orchestration_fakes.TestOrchestrationv1):

    def setUp(self):
        super(TestDeployment, self).setUp()
        self.mock_client = self.app.client_manager.orchestration
        self.config_client = self.mock_client.software_configs
        self.sd_client = self.mock_client.software_deployments