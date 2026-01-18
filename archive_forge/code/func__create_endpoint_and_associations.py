import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _create_endpoint_and_associations(self, project_id, service_id=None):
    """Create an endpoint associated with service and project."""
    if not service_id:
        service_ref = unit.new_service_ref()
        response = self.post('/services', body={'service': service_ref})
        service_id = response.result['service']['id']
    endpoint_ref = unit.new_endpoint_ref(service_id=service_id, interface='public', region_id=self.region_id)
    response = self.post('/endpoints', body={'endpoint': endpoint_ref})
    endpoint = response.result['endpoint']
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint['id']})
    return endpoint