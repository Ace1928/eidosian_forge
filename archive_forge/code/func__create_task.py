from unittest import mock
import oslo_policy.policy
from oslo_serialization import jsonutils
from glance.api import policy
from glance.tests import functional
def _create_task(self, path=None, data=None, expected_code=201):
    if not path:
        path = '/v2/tasks'
    resp = self.api_post(path, json=data)
    task = jsonutils.loads(resp.text)
    self.assertEqual(expected_code, resp.status_code)
    return task