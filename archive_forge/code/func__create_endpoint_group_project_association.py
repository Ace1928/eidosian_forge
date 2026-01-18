import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _create_endpoint_group_project_association(self, endpoint_group_id, project_id):
    url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
    self.put(url)