import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderMemberDomainAndProjectUserDomainConfigTests(object):

    def test_user_cannot_create_domain_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.put('/v3/domains/%s/config' % domain['id'], json={'config': unit.new_domain_config_ref()}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_domain_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        new_config = {'ldap': {'url': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/domains/%s/config' % domain['id'], json={'config': new_config}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_domain_group_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        new_config = {'ldap': {'url': uuid.uuid4().hex, 'user_filter': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/domains/%s/config/ldap' % domain['id'], json={'config': new_config}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_domain_config_option(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        new_config = {'url': uuid.uuid4().hex}
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.patch('/v3/domains/%s/config/ldap/url' % domain['id'], json={'config': new_config}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_domain_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.delete('/v3/domains/%s/config' % domain['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_domain_group_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.delete('/v3/domains/%s/config/ldap' % domain['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_domain_config_option(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.delete('/v3/domains/%s/config/ldap/url' % domain['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)