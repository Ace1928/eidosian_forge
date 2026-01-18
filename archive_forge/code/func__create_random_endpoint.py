import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def _create_random_endpoint(self, interface='public', parent_region_id=None):
    region = self._create_region_with_parent_id(parent_id=parent_region_id)
    service = self._create_random_service()
    ref = unit.new_endpoint_ref(service_id=service['id'], interface=interface, region_id=region.result['region']['id'])
    response = self.post('/endpoints', body={'endpoint': ref})
    return response.json['endpoint']