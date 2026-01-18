import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def _crud_policy_association_for_service_via_id(self, http_action, manager_action):
    policy_id = uuid.uuid4().hex
    service_id = uuid.uuid4().hex
    self.stub_url(http_action, ['policies', policy_id, self.manager.OS_EP_POLICY_EXT, 'services', service_id], status_code=204)
    manager_action(policy=policy_id, service=service_id)