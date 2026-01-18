import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def _assert_correct_endpoints(self, policy, endpoint_list):
    endpoint_id_list = [ep['id'] for ep in endpoint_list]
    endpoints = PROVIDERS.endpoint_policy_api.list_endpoints_for_policy(policy['id'])
    self.assertThat(endpoints, matchers.HasLength(len(endpoint_list)))
    for endpoint in endpoints:
        self.assertIn(endpoint['id'], endpoint_id_list)