import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _UserRegisteredLimitTests(object):
    """Common default functionality for all users except system admins."""

    def test_user_can_get_a_registered_limit(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
        limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        limit_id = limits[0]['id']
        with self.test_client() as c:
            r = c.get('/v3/registered_limits/%s' % limit_id, headers=self.headers)
            self.assertEqual(limit_id, r.json['registered_limit']['id'])

    def test_user_can_list_registered_limits(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
        limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        limit_id = limits[0]['id']
        with self.test_client() as c:
            r = c.get('/v3/registered_limits', headers=self.headers)
            self.assertTrue(len(r.json['registered_limits']) == 1)
            self.assertEqual(limit_id, r.json['registered_limits'][0]['id'])

    def test_user_cannot_create_registered_limits(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        create = {'registered_limits': [unit.new_registered_limit_ref(service_id=service['id'])]}
        with self.test_client() as c:
            c.post('/v3/registered_limits', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_registered_limits(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
        limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        limit_id = limits[0]['id']
        with self.test_client() as c:
            update = {'registered_limit': {'default_limit': 5}}
            c.patch('/v3/registered_limits/%s' % limit_id, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_registered_limits(self):
        service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
        registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
        limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
        limit_id = limits[0]['id']
        with self.test_client() as c:
            c.delete('/v3/registered_limits/%s' % limit_id, headers=self.headers, expected_status_code=http.client.FORBIDDEN)