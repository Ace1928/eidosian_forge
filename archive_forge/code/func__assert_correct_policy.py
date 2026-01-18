import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def _assert_correct_policy(self, endpoint, policy):
    ref = PROVIDERS.endpoint_policy_api.get_policy_for_endpoint(endpoint['id'])
    self.assertEqual(policy['id'], ref['id'])