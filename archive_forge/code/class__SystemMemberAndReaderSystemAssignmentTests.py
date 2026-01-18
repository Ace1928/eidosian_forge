import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemMemberAndReaderSystemAssignmentTests(object):

    def test_user_cannot_grant_system_assignments(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id))
        with self.test_client() as c:
            c.put('/v3/system/users/%s/roles/%s' % (user['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_revoke_system_assignments(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            c.delete('/v3/system/users/%s/roles/%s' % (user['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_grant_group_system_assignment(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(CONF.identity.default_domain_id))
        with self.test_client() as c:
            c.put('/v3/system/groups/%s/roles/%s' % (group['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_revoke_group_system_assignment(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_system_grant_for_group(group['id'], self.bootstrapper.member_role_id)
        with self.test_client() as c:
            c.delete('/v3/system/groups/%s/roles/%s' % (group['id'], self.bootstrapper.member_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)