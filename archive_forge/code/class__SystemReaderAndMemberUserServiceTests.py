import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderAndMemberUserServiceTests(object):
    """Common default functionality for system readers and system members."""

    def test_user_cannot_create_services(self):
        create = {'service': {'type': uuid.uuid4().hex, 'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/services', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_services(self):
        service = unit.new_service_ref()
        service = PROVIDERS.catalog_api.create_service(service['id'], service)
        update = {'service': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/services/%s' % service['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_services(self):
        service = unit.new_service_ref()
        service = PROVIDERS.catalog_api.create_service(service['id'], service)
        with self.test_client() as c:
            c.delete('/v3/services/%s' % service['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)