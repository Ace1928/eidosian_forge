import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def _crud_policy_association_for_service_via_obj(self, http_action, manager_action):
    policy_ref = self.new_policy_ref()
    service_ref = self.new_service_ref()
    policy = self.client.policies.resource_class(self.client.policies, policy_ref, loaded=True)
    service = self.client.services.resource_class(self.client.services, service_ref, loaded=True)
    self.stub_url(http_action, ['policies', policy_ref['id'], self.manager.OS_EP_POLICY_EXT, 'services', service_ref['id']], status_code=204)
    manager_action(policy=policy, service=service)