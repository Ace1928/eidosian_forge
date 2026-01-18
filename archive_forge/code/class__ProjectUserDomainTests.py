import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import domain as dp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _ProjectUserDomainTests(object):

    def test_user_can_get_a_domain(self):
        with self.test_client() as c:
            r = c.get('/v3/domains/%s' % self.domain_id, headers=self.headers)
            self.assertEqual(self.domain_id, r.json['domain']['id'])

    def test_user_cannot_get_a_domain_they_are_not_authorized_to_access(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.get('/v3/domains/%s' % domain['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_domains(self):
        with self.test_client() as c:
            c.get('/v3/domains', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_domains_by_name(self):
        domain_name = uuid.uuid4().hex
        domain = unit.new_domain_ref(name=domain_name)
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.get('/v3/domains?name=%s' % domain_name, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_filter_domains_by_enabled(self):
        with self.test_client() as c:
            c.get('/v3/domains?enabled=true', headers=self.headers, expected_status_code=http.client.FORBIDDEN)
            c.get('/v3/domains?enabled=false', headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_a_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        update = {'domain': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/domains/%s' % domain['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_create_a_domain(self):
        create = {'domain': {'name': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/domains', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_a_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            update = {'domain': {'enabled': False}}
            path = '/v3/domains/%s' % domain['id']
            c.patch(path, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)
            c.delete(path, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_non_existant_domain_forbidden(self):
        with self.test_client() as c:
            c.get('/v3/domains/%s' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)