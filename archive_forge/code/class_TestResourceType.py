from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource_type
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resource_types
class TestResourceType(orchestration_fakes.TestOrchestrationv1):

    def setUp(self):
        super(TestResourceType, self).setUp()
        self.mock_client = self.app.client_manager.orchestration