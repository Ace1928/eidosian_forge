import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _UserEC2CredentialTests(object):

    def test_user_can_get_their_ec2_credentials(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=project['id'])
        with self.test_client() as c:
            r = c.post('/v3/users/%s/credentials/OS-EC2' % self.user_id, json={'tenant_id': project['id']}, headers=self.headers)
            credential_id = r.json['credential']['access']
            path = '/v3/users/%s/credentials/OS-EC2/%s' % (self.user_id, credential_id)
            r = c.get(path, headers=self.headers)
            self.assertEqual(self.user_id, r.json['credential']['user_id'])

    def test_user_can_list_their_ec2_credentials(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=project['id'])
        with self.test_client() as c:
            c.post('/v3/users/%s/credentials/OS-EC2' % self.user_id, json={'tenant_id': project['id']}, headers=self.headers)
            path = '/v3/users/%s/credentials/OS-EC2' % self.user_id
            r = c.get(path, headers=self.headers)
            for credential in r.json['credentials']:
                self.assertEqual(self.user_id, credential['user_id'])

    def test_user_create_their_ec2_credentials(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=project['id'])
        with self.test_client() as c:
            c.post('/v3/users/%s/credentials/OS-EC2' % self.user_id, json={'tenant_id': project['id']}, headers=self.headers, expected_status_code=http.client.CREATED)

    def test_user_delete_their_ec2_credentials(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=project['id'])
        with self.test_client() as c:
            r = c.post('/v3/users/%s/credentials/OS-EC2' % self.user_id, json={'tenant_id': project['id']}, headers=self.headers)
            credential_id = r.json['credential']['access']
            c.delete('/v3/users/%s/credentials/OS-EC2/%s' % (self.user_id, credential_id), headers=self.headers)

    def test_user_cannot_create_ec2_credentials_for_others(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            c.post('/v3/users/%s/credentials/OS-EC2' % user['id'], json={'tenant_id': project['id']}, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_ec2_credentials_for_others(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_password = user['password']
        user = PROVIDERS.identity_api.create_user(user)
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        user_auth = self.build_authentication_request(user_id=user['id'], password=user_password, project_id=project['id'])
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=user_auth)
            token_id = r.headers['X-Subject-Token']
            headers = {'X-Auth-Token': token_id}
            r = c.post('/v3/users/%s/credentials/OS-EC2' % user['id'], json={'tenant_id': project['id']}, headers=headers)
            credential_id = r.json['credential']['access']
            c.delete('/v3/users/%s/credentials/OS-EC2/%s' % (self.user_id, credential_id), headers=self.headers, expected_status_code=http.client.FORBIDDEN)