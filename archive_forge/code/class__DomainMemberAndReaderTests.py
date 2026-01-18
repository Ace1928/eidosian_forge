import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainMemberAndReaderTests(object):

    def test_cannot_create_grant_for_user_on_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        with self.test_client() as c:
            c.put('/v3/projects/%s/users/%s/roles/%s' % (project['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_create_grant_for_user_on_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.put('/v3/domains/%s/users/%s/roles/%s' % (domain['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_create_grant_for_group_on_project(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            c.put('/v3/projects/%s/groups/%s/roles/%s' % (project['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_create_grant_for_group_on_domain(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.put('/v3/domains/%s/groups/%s/roles/%s' % (domain['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_revoke_grant_from_user_on_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            c.delete('/v3/projects/%s/users/%s/roles/%s' % (project['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_revoke_grant_from_user_on_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], domain_id=domain['id'])
        with self.test_client() as c:
            c.delete('/v3/domains/%s/users/%s/roles/%s' % (domain['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_revoke_grant_from_group_on_project(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], project_id=project['id'])
        with self.test_client() as c:
            c.delete('/v3/projects/%s/groups/%s/roles/%s' % (project['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_cannot_revoke_grant_from_group_on_domain(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=self.domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], domain_id=domain['id'])
        with self.test_client() as c:
            c.delete('/v3/domains/%s/groups/%s/roles/%s' % (domain['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)