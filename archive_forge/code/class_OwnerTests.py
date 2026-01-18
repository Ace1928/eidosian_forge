import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class OwnerTests(_TestAppCredBase, common_auth.AuthTestMixin, _SystemUserAndOwnerTests):

    def setUp(self):
        super(OwnerTests, self).setUp()
        self.loadapp()
        self.policy_file = self.useFixture(temporaryfile.SecureTempFile())
        self.policy_file_name = self.policy_file.file_name
        self.useFixture(ksfixtures.Policy(self.config_fixture, policy_file=self.policy_file_name))
        self._override_policy()
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        self.user_id = self.app_cred_user_id
        auth = self.build_authentication_request(user_id=self.user_id, password=self.app_cred_user_password, project_id=self.app_cred_project_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}

    def test_create_application_credential_by_owner(self):
        app_cred_body = {'application_credential': unit.new_application_credential_ref()}
        with self.test_client() as c:
            c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers=self.headers)

    def test_owner_can_delete_application_credential(self):
        self._test_delete_application_credential()

    def test_user_cannot_lookup_application_credential_for_another_user(self):
        another_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        another_user_id = PROVIDERS.identity_api.create_user(another_user)['id']
        auth = self.build_authentication_request(user_id=another_user_id, password=another_user['password'])
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            another_user_token = r.headers['X-Subject-Token']
        app_cred = self._create_application_credential()
        with self.test_client() as c:
            c.get('/v3/users/%s/application_credentials/%s' % (another_user_id, app_cred['id']), expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': another_user_token})

    def test_user_cannot_delete_application_credential_for_another_user(self):
        another_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        another_user_id = PROVIDERS.identity_api.create_user(another_user)['id']
        auth = self.build_authentication_request(user_id=another_user_id, password=another_user['password'])
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            another_user_token = r.headers['X-Subject-Token']
        app_cred = self._create_application_credential()
        with self.test_client() as c:
            c.delete('/v3/users/%s/application_credentials/%s' % (another_user_id, app_cred['id']), expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': another_user_token})