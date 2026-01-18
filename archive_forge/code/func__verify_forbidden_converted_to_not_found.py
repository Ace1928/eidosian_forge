from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
    headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
    resp = self.api_request(method, path, headers=headers, json=json)
    self.assertEqual(404, resp.status_code)