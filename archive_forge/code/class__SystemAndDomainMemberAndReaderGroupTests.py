import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import group as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemAndDomainMemberAndReaderGroupTests(object):
    """Common default functionality for system readers and system members."""

    def test_user_cannot_create_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        create = {'group': {'name': uuid.uuid4().hex, 'domain_id': domain['id']}}
        with self.test_client() as c:
            c.post('/v3/groups', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        update = {'group': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/groups/%s' % group['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        with self.test_client() as c:
            c.delete('/v3/groups/%s' % group['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_add_users_to_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        with self.test_client() as c:
            c.put('/v3/groups/%s/users/%s' % (group['id'], user['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_remove_users_from_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.delete('/v3/groups/%s/users/%s' % (group['id'], user['id']), headers=self.headers, expected_status_code=http.client.FORBIDDEN)