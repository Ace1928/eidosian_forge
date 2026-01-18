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
class _SystemUserGroupTests(object):
    """Common default functionality for all system users."""

    def test_user_can_list_groups(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        with self.test_client() as c:
            r = c.get('/v3/groups', headers=self.headers)
            self.assertEqual(1, len(r.json['groups']))
            self.assertEqual(group['id'], r.json['groups'][0]['id'])

    def test_user_can_get_a_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        with self.test_client() as c:
            r = c.get('/v3/groups/%s' % group['id'], headers=self.headers)
            self.assertEqual(group['id'], r.json['group']['id'])

    def test_user_can_list_group_members(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            r = c.get('/v3/groups/%s/users' % group['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['users']))
            self.assertEqual(user['id'], r.json['users'][0]['id'])

    def test_user_can_list_groups_for_other_users(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            r = c.get('/v3/users/%s/groups' % user['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['groups']))
            self.assertEqual(group['id'], r.json['groups'][0]['id'])

    def test_user_can_check_if_user_in_group(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain['id']))
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain['id']))
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        with self.test_client() as c:
            c.get('/v3/groups/%s/users/%s' % (group['id'], user['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_cannot_get_non_existent_group_not_found(self):
        with self.test_client() as c:
            c.get('/v3/groups/%s' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.NOT_FOUND)