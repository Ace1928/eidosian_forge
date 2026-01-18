import json
import uuid
import http.client
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
class PolicyTestCase(test_v3.RestfulTestCase):
    """Test policy CRUD."""

    def setUp(self):
        super(PolicyTestCase, self).setUp()
        self.policy = unit.new_policy_ref()
        self.policy_id = self.policy['id']
        PROVIDERS.policy_api.create_policy(self.policy_id, self.policy.copy())

    def test_create_policy(self):
        """Call ``POST /policies``."""
        ref = unit.new_policy_ref()
        r = self.post('/policies', body={'policy': ref})
        return self.assertValidPolicyResponse(r, ref)

    def test_list_head_policies(self):
        """Call ``GET & HEAD /policies``."""
        resource_url = '/policies'
        r = self.get(resource_url)
        self.assertValidPolicyListResponse(r, ref=self.policy)
        self.head(resource_url, expected_status=http.client.OK)

    def test_get_head_policy(self):
        """Call ``GET & HEAD /policies/{policy_id}``."""
        resource_url = '/policies/%(policy_id)s' % {'policy_id': self.policy_id}
        r = self.get(resource_url)
        self.assertValidPolicyResponse(r, self.policy)
        self.head(resource_url, expected_status=http.client.OK)

    def test_update_policy(self):
        """Call ``PATCH /policies/{policy_id}``."""
        self.policy['blob'] = json.dumps({'data': uuid.uuid4().hex})
        r = self.patch('/policies/%(policy_id)s' % {'policy_id': self.policy_id}, body={'policy': self.policy})
        self.assertValidPolicyResponse(r, self.policy)

    def test_delete_policy(self):
        """Call ``DELETE /policies/{policy_id}``."""
        self.delete('/policies/%(policy_id)s' % {'policy_id': self.policy_id})