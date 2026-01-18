import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserEndpointTests(object):
    """Common default functionality for all system users."""

    def test_user_can_list_endpoints(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            r = c.get('/v3/endpoints', headers=self.headers)
            endpoints = []
            for endpoint in r.json['endpoints']:
                endpoints.append(endpoint['id'])
            self.assertIn(endpoint['id'], endpoints)

    def test_user_can_get_an_endpoint(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
        endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        with self.test_client() as c:
            c.get('/v3/endpoints/%s' % endpoint['id'], headers=self.headers)